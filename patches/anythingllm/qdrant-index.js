const { QdrantClient } = require("@qdrant/js-client-rest");
const { TextSplitter } = require("../../TextSplitter");
const { SystemSettings } = require("../../../models/systemSettings");
const { storeVectorResult, cachedVectorInformation } = require("../../files");
const { v4: uuidv4 } = require("uuid");
const { toChunks, getEmbeddingEngineSelection } = require("../../helpers");
const { sourceIdentifier } = require("../../chats");
const { VectorDatabase } = require("../base");

/**
 * CBDU project patch: calls the dedicated GPU reranker (TEI serving
 * BAAI/bge-reranker-v2-m3, see docker-compose.yml's tei-rerank service)
 * instead of AnythingLLM's bundled CPU-only NativeEmbeddingReranker, which
 * blocked Node's event loop for 5-8s per query -- long enough to trip
 * vLLM's default 5s keep-alive timeout and cause intermittent
 * "Connection error" failures on the follow-up completion request (see
 * VLLM_HTTP_TIMEOUT_KEEP_ALIVE in docker-compose.yml for that half of the
 * fix). TEI runs the cross-encoder on GPU 1 in tens of ms instead.
 * @param {string} query
 * @param {{text: string}[]} candidates
 * @returns {Promise<Array>} candidates with a `rerank_score` field, sorted descending
 */
async function cbduTeiRerank(query, candidates) {
  const endpoint = process.env.CBDU_RERANK_ENDPOINT;
  if (!endpoint) throw new Error("CBDU_RERANK_ENDPOINT not set");
  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, texts: candidates.map((c) => c.text) }),
  });
  if (!res.ok) throw new Error(`TEI rerank HTTP ${res.status}`);
  const ranked = await res.json(); // [{index, score}, ...] sorted desc by score
  return ranked.map(({ index, score }) => ({
    ...candidates[index],
    rerank_score: score,
  }));
}

class QDrant extends VectorDatabase {
  constructor() {
    super();
  }

  get name() {
    return "QDrant";
  }

  async connect() {
    if (process.env.VECTOR_DB !== "qdrant")
      throw new Error("QDrant::Invalid ENV settings");

    const client = new QdrantClient({
      url: process.env.QDRANT_ENDPOINT,
      ...(process.env.QDRANT_API_KEY
        ? { apiKey: process.env.QDRANT_API_KEY }
        : {}),
    });

    const isAlive = (await client.api("cluster")?.clusterStatus())?.ok || false;
    if (!isAlive)
      throw new Error(
        "QDrant::Invalid Heartbeat received - is the instance online?"
      );

    return { client };
  }

  async heartbeat() {
    await this.connect();
    return { heartbeat: Number(new Date()) };
  }

  async totalVectors() {
    const { client } = await this.connect();
    const { collections } = await client.getCollections();
    var totalVectors = 0;
    for (const collection of collections) {
      if (!collection || !collection.name) continue;
      totalVectors +=
        (await this.namespace(client, collection.name))?.vectorCount || 0;
    }
    return totalVectors;
  }

  async namespaceCount(_namespace = null) {
    const { client } = await this.connect();
    const namespace = await this.namespace(client, _namespace);
    return namespace?.vectorCount || 0;
  }

  async similarityResponse({
    client,
    namespace,
    queryVector,
    similarityThreshold = 0.25,
    topN = 4,
    filterIdentifiers = [],
  }) {
    const result = {
      contextTexts: [],
      sourceDocuments: [],
      scores: [],
    };

    const responses = await client.search(namespace, {
      vector: queryVector,
      limit: topN,
      with_payload: true,
    });

    responses.forEach((response) => {
      if (response.score < similarityThreshold) return;
      if (filterIdentifiers.includes(sourceIdentifier(response?.payload))) {
        this.logger(
          "QDrant: A source was filtered from context as it's parent document is pinned."
        );
        return;
      }

      result.contextTexts.push(response?.payload?.text || "");
      result.sourceDocuments.push({
        ...(response?.payload || {}),
        id: response.id,
        score: response.score,
      });
      result.scores.push(response.score);
    });

    return result;
  }

  /**
   * CBDU project patch: Qdrant has no native reranking support upstream --
   * `performSimilaritySearch` below never even accepted a `rerank` param,
   * so it silently dropped `workspace.vectorSearchMode === "rerank"`
   * (set from chats/stream.js etc. uniformly for every provider) and always
   * did plain cosine similarity. Pulls a wide candidate pool via plain
   * vector search, reranks it via the GPU-hosted TEI cross-encoder
   * (cbduTeiRerank above), then truncates to topN.
   */
  async rerankedSimilarityResponse({
    client,
    namespace,
    query,
    queryVector,
    topN = 4,
    similarityThreshold = 0.25,
    filterIdentifiers = [],
  }) {
    const result = { contextTexts: [], sourceDocuments: [], scores: [] };

    // CBDU project: true over-fetch-then-shrink -- pull a wide candidate
    // pool via plain cosine similarity so the cross-encoder has enough to
    // work with to rescue chunks cosine underrated, then shrink to the
    // topN actually fed into the LLM's context. Pool size comes from .env
    // (CBDU_RERANK_POOL_SIZE, default 200) rather than being topN-scaled;
    // only ever raised to topN itself if topN somehow exceeds that.
    const poolSize = parseInt(process.env.CBDU_RERANK_POOL_SIZE, 10) || 200;
    const searchLimit = Math.max(topN, poolSize);
    const responses = await client.search(namespace, {
      vector: queryVector,
      limit: searchLimit,
      with_payload: true,
    });

    const candidates = responses
      .filter((r) => r.score >= similarityThreshold)
      .filter((r) => !filterIdentifiers.includes(sourceIdentifier(r?.payload)))
      .map((r) => ({
        ...(r?.payload || {}),
        id: r.id,
        _origScore: r.score,
        text: r?.payload?.text || "",
      }));
    if (!candidates.length) return result;

    let ranked;
    try {
      ranked = await cbduTeiRerank(query, candidates);
    } catch (e) {
      // TEI unreachable/erroring -- fall back to plain cosine order rather
      // than fail the whole request. Deliberately NOT falling back to the
      // local CPU reranker: that's the 5-8s-per-query path whose latency
      // caused the keep-alive connection errors this patch exists to fix.
      this.logger(e);
      this.logger(
        "rerankedSimilarityResponse: TEI rerank failed, falling back to cosine order",
        e.message
      );
      ranked = [...candidates].sort((a, b) => b._origScore - a._origScore);
    }

    // CBDU project: bge-reranker-v2-m3's raw scores are on a completely
    // different scale than cosine similarity (confirmed empirically -- even
    // a genuinely top match scores ~0.12-0.18, nowhere near
    // similarityThreshold's 0-1 cosine semantics). Using it to reapply that
    // threshold would incorrectly filter out the best-matching chunks, so
    // it's used ONLY to reorder candidates. The score attached to each
    // result (and shown as "% dopasowanie" in the GUI) stays the original
    // cosine similarity, which already satisfies similarityThreshold by
    // construction (candidates were filtered on it above).
    ranked.slice(0, topN).forEach((item) => {
      const { text, rerank_score, _origScore, ...rest } = item;
      result.contextTexts.push(text);
      result.sourceDocuments.push({ ...rest, score: _origScore });
      result.scores.push(_origScore);
    });

    return result;
  }

  async namespace(client, namespace = null) {
    if (!namespace) throw new Error("No namespace value provided.");
    const collection = await client.getCollection(namespace).catch(() => null);
    if (!collection) return null;

    return {
      name: namespace,
      ...collection,
      vectorCount: (await client.count(namespace, { exact: true })).count,
    };
  }

  async hasNamespace(namespace = null) {
    if (!namespace) return false;
    const { client } = await this.connect();
    return await this.namespaceExists(client, namespace);
  }

  async namespaceExists(client, namespace = null) {
    if (!namespace) throw new Error("No namespace value provided.");
    const collection = await client.getCollection(namespace).catch((e) => {
      this.logger("namespaceExists", e.message);
      return null;
    });
    return !!collection;
  }

  async deleteVectorsInNamespace(client, namespace = null) {
    await client.deleteCollection(namespace);
    return true;
  }

  // QDrant requires a dimension aspect for collection creation
  // we pass this in from the first chunk to infer the dimensions like other
  // providers do.
  async getOrCreateCollection(client, namespace, dimensions = null) {
    if (await this.namespaceExists(client, namespace)) {
      return await client.getCollection(namespace);
    }
    if (!dimensions)
      throw new Error(
        `Qdrant:getOrCreateCollection Unable to infer vector dimension from input. Open an issue on GitHub for support.`
      );
    await client.createCollection(namespace, {
      vectors: {
        size: dimensions,
        distance: "Cosine",
      },
    });
    return await client.getCollection(namespace);
  }

  async addDocumentToNamespace(
    namespace,
    documentData = {},
    fullFilePath = null,
    skipCache = false
  ) {
    const { DocumentVectors } = require("../../../models/vectors");
    try {
      let vectorDimension = null;
      const { pageContent, docId, ...metadata } = documentData;
      if (!pageContent || pageContent.length == 0) return false;

      this.logger("Adding new vectorized document into namespace", namespace);
      if (!skipCache) {
        const cacheResult = await cachedVectorInformation(fullFilePath);
        if (cacheResult.exists) {
          const { client } = await this.connect();
          const { chunks } = cacheResult;
          const documentVectors = [];
          vectorDimension =
            chunks[0][0]?.vector?.length ??
            chunks[0][0]?.values?.length ??
            null;

          const collection = await this.getOrCreateCollection(
            client,
            namespace,
            vectorDimension
          );
          if (!collection)
            throw new Error("Failed to create new QDrant collection!", {
              namespace,
            });

          for (const chunk of chunks) {
            const submission = {
              ids: [],
              vectors: [],
              payloads: [],
            };

            // Before sending to Qdrant and saving the records to our db
            // we need to assign the id of each chunk that is stored in the cached file.
            // The id property must be defined or else it will be unable to be managed by ALLM.
            chunk.forEach((chunk) => {
              const id = uuidv4();
              if (chunk?.payload?.hasOwnProperty("id")) {
                const { id: _id, ...payload } = chunk.payload;
                documentVectors.push({ docId, vectorId: id });
                submission.ids.push(id);
                submission.vectors.push(chunk.vector);
                submission.payloads.push(payload);
              } else {
                console.error(
                  "The 'id' property is not defined in chunk.payload - it will be omitted from being inserted in QDrant collection."
                );
              }
            });

            const additionResult = await client.upsert(namespace, {
              wait: true,
              batch: { ...submission },
            });
            if (additionResult?.status !== "completed")
              throw new Error("Error embedding into QDrant", additionResult);
          }

          await DocumentVectors.bulkInsert(documentVectors);
          return { vectorized: true, error: null };
        }
      }

      // If we are here then we are going to embed and store a novel document.
      // We have to do this manually as opposed to using LangChains `Qdrant.fromDocuments`
      // because we then cannot atomically control our namespace to granularly find/remove documents
      // from vectordb.
      const EmbedderEngine = getEmbeddingEngineSelection();
      const textSplitter = new TextSplitter({
        chunkSize: TextSplitter.determineMaxChunkSize(
          await SystemSettings.getValueOrFallback({
            label: "text_splitter_chunk_size",
          }),
          EmbedderEngine?.embeddingMaxChunkLength
        ),
        chunkOverlap: await SystemSettings.getValueOrFallback(
          { label: "text_splitter_chunk_overlap" },
          20
        ),
        chunkHeaderMeta: TextSplitter.buildHeaderMeta(metadata),
        chunkPrefix: EmbedderEngine?.embeddingPrefix,
      });
      const textChunks = await textSplitter.splitText(pageContent);

      this.logger("Snippets created from document:", textChunks.length);
      const documentVectors = [];
      const vectors = [];
      const vectorValues = await EmbedderEngine.embedChunks(textChunks);
      const submission = {
        ids: [],
        vectors: [],
        payloads: [],
      };

      if (!!vectorValues && vectorValues.length > 0) {
        for (const [i, vector] of vectorValues.entries()) {
          if (!vectorDimension) vectorDimension = vector.length;
          const vectorRecord = {
            id: uuidv4(),
            vector: vector,
            // [DO NOT REMOVE]
            // LangChain will be unable to find your text if you embed manually and dont include the `text` key.
            // https://github.com/hwchase17/langchainjs/blob/2def486af734c0ca87285a48f1a04c057ab74bdf/langchain/src/vectorstores/pinecone.ts#L64
            payload: { ...metadata, text: textChunks[i] },
          };

          submission.ids.push(vectorRecord.id);
          submission.vectors.push(vectorRecord.vector);
          submission.payloads.push(vectorRecord.payload);

          vectors.push(vectorRecord);
          documentVectors.push({ docId, vectorId: vectorRecord.id });
        }
      } else {
        throw new Error(
          "Could not embed document chunks! This document will not be recorded."
        );
      }

      const { client } = await this.connect();
      const collection = await this.getOrCreateCollection(
        client,
        namespace,
        vectorDimension
      );
      if (!collection)
        throw new Error("Failed to create new QDrant collection!", {
          namespace,
        });

      if (vectors.length > 0) {
        const chunks = [];

        this.logger("Inserting vectorized chunks into QDrant collection.");
        for (const chunk of toChunks(vectors, 500)) {
          const batchIds = [],
            batchVectors = [],
            batchPayloads = [];
          chunks.push(chunk);
          chunk.forEach((v) => {
            batchIds.push(v.id);
            batchVectors.push(v.vector);
            batchPayloads.push(v.payload);
          });

          const additionResult = await client.upsert(namespace, {
            wait: true,
            batch: {
              ids: batchIds,
              vectors: batchVectors,
              payloads: batchPayloads,
            },
          });
          if (additionResult?.status !== "completed")
            throw new Error("Error embedding into QDrant", additionResult);
        }

        await storeVectorResult(chunks, fullFilePath);
      }

      await DocumentVectors.bulkInsert(documentVectors);
      return { vectorized: true, error: null };
    } catch (e) {
      this.logger("addDocumentToNamespace", e.message);
      return { vectorized: false, error: e.message };
    }
  }

  async deleteDocumentFromNamespace(namespace, docId) {
    const { DocumentVectors } = require("../../../models/vectors");
    const { client } = await this.connect();
    if (!(await this.namespaceExists(client, namespace))) return;

    const knownDocuments = await DocumentVectors.where({ docId });
    if (knownDocuments.length === 0) return;

    const vectorIds = knownDocuments.map((doc) => doc.vectorId);
    await client.delete(namespace, {
      wait: true,
      points: vectorIds,
    });

    const indexes = knownDocuments.map((doc) => doc.id);
    await DocumentVectors.deleteIds(indexes);
    return true;
  }

  async performSimilaritySearch({
    namespace = null,
    input = "",
    LLMConnector = null,
    similarityThreshold = 0.25,
    topN = 4,
    filterIdentifiers = [],
    rerank = false,
  }) {
    if (!namespace || !input || !LLMConnector)
      throw new Error("Invalid request to performSimilaritySearch.");

    const { client } = await this.connect();
    if (!(await this.namespaceExists(client, namespace))) {
      return {
        contextTexts: [],
        sources: [],
        message: "Invalid query - no documents found for workspace!",
      };
    }

    const queryVector = await LLMConnector.embedTextInput(input);
    const { contextTexts, sourceDocuments } = rerank
      ? await this.rerankedSimilarityResponse({
          client,
          namespace,
          query: input,
          queryVector,
          similarityThreshold,
          topN,
          filterIdentifiers,
        })
      : await this.similarityResponse({
          client,
          namespace,
          queryVector,
          similarityThreshold,
          topN,
          filterIdentifiers,
        });

    const sources = sourceDocuments.map((metadata, i) => {
      return { ...metadata, text: contextTexts[i] };
    });
    return {
      contextTexts,
      sources: this.curateSources(sources),
      message: false,
    };
  }

  async "namespace-stats"(reqBody = {}) {
    const { namespace = null } = reqBody;
    if (!namespace) throw new Error("namespace required");
    const { client } = await this.connect();
    if (!(await this.namespaceExists(client, namespace)))
      throw new Error("Namespace by that name does not exist.");
    const stats = await this.namespace(client, namespace);
    return stats
      ? stats
      : { message: "No stats were able to be fetched from DB for namespace" };
  }

  async "delete-namespace"(reqBody = {}) {
    const { namespace = null } = reqBody;
    const { client } = await this.connect();
    if (!(await this.namespaceExists(client, namespace)))
      throw new Error("Namespace by that name does not exist.");

    const details = await this.namespace(client, namespace);
    await this.deleteVectorsInNamespace(client, namespace);
    return {
      message: `Namespace ${namespace} was deleted along with ${details?.vectorCount} vectors.`,
    };
  }

  async reset() {
    const { client } = await this.connect();
    const response = await client.getCollections();
    for (const collection of response.collections) {
      await client.deleteCollection(collection.name);
    }
    return { reset: true };
  }

  curateSources(sources = []) {
    const documents = [];
    for (const source of sources) {
      if (Object.keys(source).length > 0) {
        const metadata = source.hasOwnProperty("metadata")
          ? source.metadata
          : source;
        documents.push({
          ...metadata,
        });
      }
    }

    return documents;
  }
}

module.exports.QDrant = QDrant;

const { v4 } = require("uuid");
const { writeToServerDocuments } = require("../utils/files");
const { tokenizeString } = require("../utils/tokenizer");
const { default: slugify } = require("slugify");

// Will remove the last .extension from the input
// and stringify the input + move to lowercase.
function stripAndSlug(input) {
  if (!input.includes(".")) return slugify(input, { lower: true });
  return slugify(input.split(".").slice(0, -1).join("-"), { lower: true });
}

const METADATA_KEYS = {
  possible: {
    url: ({ url, title }) => {
      let validUrl;
      try {
        const u = new URL(url);
        validUrl = ["https:", "http:"].includes(u.protocol);
      } catch {}

      if (validUrl) return `web://${url.toLowerCase()}.website`;
      return `file://${stripAndSlug(title)}.txt`;
    },
    // CBDU project patch: stock behavior force-slugified every title into
    // "some-title.txt", which is what the frontend's citation header (kg()
    // in the client bundle) displays for our chunkSource prefix -- so
    // citations showed an ugly filename-looking slug instead of the real,
    // human-readable document title. We already send a clean title string
    // from ingest.py; just pass it through untouched.
    title: ({ title }) => (typeof title === "string" && title.trim() ? title.trim() : "untitled"),
    docAuthor: ({ docAuthor }) => {
      return typeof docAuthor === "string" ? docAuthor : "no author specified";
    },
    description: ({ description }) => {
      return typeof description === "string"
        ? description
        : "no description found";
    },
    docSource: ({ docSource }) => {
      return typeof docSource === "string" ? docSource : "no source set";
    },
    chunkSource: ({ chunkSource, title }) => {
      return typeof chunkSource === "string"
        ? chunkSource
        : `${stripAndSlug(title)}.txt`;
    },
    published: ({ published }) => {
      if (isNaN(Number(published))) return new Date().toLocaleString();
      return new Date(Number(published)).toLocaleString();
    },
  },
};

async function processRawText(textContent, metadata) {
  console.log(`-- Working Raw Text doc ${metadata.title} --`);
  if (!textContent || textContent.length === 0) {
    return {
      success: false,
      reason: "textContent was empty - nothing to process.",
      documents: [],
    };
  }

  const data = {
    id: v4(),
    url: METADATA_KEYS.possible.url(metadata),
    title: METADATA_KEYS.possible.title(metadata),
    docAuthor: METADATA_KEYS.possible.docAuthor(metadata),
    description: METADATA_KEYS.possible.description(metadata),
    docSource: METADATA_KEYS.possible.docSource(metadata),
    chunkSource: METADATA_KEYS.possible.chunkSource(metadata),
    published: METADATA_KEYS.possible.published(metadata),
    wordCount: textContent.split(" ").length,
    pageContent: textContent,
    token_count_estimate: tokenizeString(textContent),
  };

  const document = writeToServerDocuments({
    data,
    filename: `raw-${stripAndSlug(metadata.title)}-${data.id}`,
  });
  console.log(
    `[SUCCESS]: Raw text and metadata saved & ready for embedding.\n`
  );
  return { success: true, reason: null, documents: [document] };
}

module.exports = { processRawText };

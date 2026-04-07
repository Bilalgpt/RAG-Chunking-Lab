import { useState } from 'react'
import { indexDocument } from '../utils/api'

const SAMPLE_DOCS = [
  { label: 'Berlin',       file: 'wikipedia_berlin.txt' },
  { label: 'Medical',      file: 'medical_guidelines.txt' },
  { label: 'Legal',        file: 'legal_contract.txt' },
  { label: 'Product Docs', file: 'product_docs.txt' },
]

const CHUNK_COLORS = [
  'bg-indigo-500/20', 'bg-emerald-500/20', 'bg-amber-500/20',
  'bg-rose-500/20', 'bg-violet-500/20',
]

function ChunkHighlight({ text, chunks }) {
  if (!chunks || chunks.length === 0) {
    return <p className="text-gray-400 text-xs font-mono whitespace-pre-wrap">{text}</p>
  }
  return (
    <div className="font-mono text-xs text-gray-300 leading-relaxed whitespace-pre-wrap">
      {chunks.map((chunk, i) => (
        <mark key={i} className={`${CHUNK_COLORS[i % 5]} rounded-sm px-0.5 text-gray-200`}>
          {chunk.text}
        </mark>
      ))}
    </div>
  )
}

export default function DocumentPanel({ selectedTechnique, llmConfig, onIndexed }) {
  const [tab, setTab] = useState('sample')
  const [docText, setDocText] = useState('')
  const [activeDoc, setActiveDoc] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [chunks, setChunks] = useState(null)
  const [indexed, setIndexed] = useState(false)

  const canIndex = selectedTechnique && docText.trim().length > 0

  async function loadSample(doc) {
    try {
      const res = await fetch(`/sample_docs/${doc.file}`)
      const text = await res.text()
      setDocText(text)
      setActiveDoc(doc.label)
      setChunks(null)
      setIndexed(false)
    } catch {
      setError('Failed to load sample document.')
    }
  }

  async function handleIndex() {
    if (!canIndex) return
    setLoading(true)
    setError(null)
    const collectionName = `${selectedTechnique}_${Date.now()}`
    try {
      const payload = {
        document_text: docText,
        technique: selectedTechnique,
        params: {},
        collection_name: collectionName,
        llm_provider: llmConfig.provider,
        api_key: llmConfig.apiKey,
      }
      const result = await indexDocument(payload)
      setChunks(result.chunks || [])
      setIndexed(true)
      onIndexed(collectionName, result, result.chunks || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-xl border border-gray-800">
      {/* Tabs */}
      <div className="flex border-b border-gray-800">
        {['sample', 'paste'].map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2.5 text-sm font-medium transition-all duration-200 ${
              tab === t
                ? 'text-indigo-400 border-b-2 border-indigo-500'
                : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            {t === 'sample' ? 'Sample Documents' : 'Paste Your Own'}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex flex-col flex-1 overflow-hidden p-3 gap-2">
        {tab === 'sample' ? (
          <div className="flex gap-2 flex-wrap">
            {SAMPLE_DOCS.map((doc) => (
              <button
                key={doc.file}
                onClick={() => loadSample(doc)}
                className={`px-3 py-1.5 rounded-lg text-sm border transition-all duration-200 ${
                  activeDoc === doc.label
                    ? 'bg-indigo-500/20 border-indigo-500 text-indigo-300'
                    : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-indigo-500/50 hover:text-gray-200'
                }`}
              >
                {doc.label}
              </button>
            ))}
          </div>
        ) : (
          <textarea
            value={docText}
            onChange={(e) => { setDocText(e.target.value); setChunks(null) }}
            placeholder="Paste your document text here…"
            className="flex-1 bg-gray-800 text-gray-200 text-xs font-mono rounded-lg p-3 border border-gray-700 focus:outline-none focus:border-indigo-500 resize-none transition-all duration-200"
          />
        )}

        {/* Document preview / chunk highlight */}
        {docText && (
          <div className="flex-1 overflow-y-auto bg-gray-800/50 rounded-lg p-3 border border-gray-700 min-h-0">
            <ChunkHighlight text={docText} chunks={chunks} />
          </div>
        )}

        {error && <p className="text-rose-400 text-xs">{error}</p>}

        {/* Index / Re-index button */}
        <div className="flex gap-2">
          <button
            onClick={handleIndex}
            disabled={!canIndex || loading}
            className={`flex-1 mt-1 py-2 rounded-lg text-sm font-semibold transition-all duration-200 ${
              canIndex && !loading
                ? 'bg-indigo-600 hover:bg-indigo-500 text-white'
                : 'bg-gray-700 text-gray-500 cursor-not-allowed'
            }`}
          >
            {loading ? 'Indexing…' : selectedTechnique ? `Index with ${selectedTechnique.replace(/_/g, ' ')}` : 'Select a technique first'}
          </button>
          {indexed && !loading && (
            <button
              onClick={() => { setChunks(null); setIndexed(false) }}
              className="mt-1 px-3 py-2 rounded-lg text-sm border border-gray-700 text-gray-400 hover:border-indigo-500/50 hover:text-gray-200 transition-all duration-200"
              title="Clear chunk highlights"
            >
              ↺
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

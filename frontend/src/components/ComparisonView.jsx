import { useState } from 'react'
import { compareDocuments } from '../utils/api'

const SLUGS = ['fixed_size', 'recursive', 'semantic', 'hierarchical', 'late_chunking', 'contextual', 'proposition']

function ResultPanel({ label, result, collection }) {
  const color = label === 'Left' ? 'indigo' : 'emerald'
  return (
    <div className="flex-1 bg-gray-900 rounded-xl border border-gray-800 p-3 space-y-2 min-w-0">
      <div className={`text-${color}-400 font-semibold text-sm`}>{label}</div>
      {!collection && <p className="text-gray-500 text-xs">Not indexed — select a technique and index first via the main panel.</p>}
      {collection && !result && <p className="text-gray-500 text-xs italic">Run a query to see results.</p>}
      {result && (
        <>
          <div className="text-gray-200 text-sm leading-relaxed">{result.answer}</div>
          <div className="text-gray-500 text-xs">{result.latency_ms} ms · {result.retrieved_chunks?.length} chunks</div>
          <div className="space-y-1 mt-1">
            {result.retrieved_chunks?.slice(0, 3).map((c, i) => (
              <div key={i} className="bg-gray-800 rounded-lg p-2 text-xs font-mono text-gray-400 line-clamp-2">
                {c.text?.slice(0, 120)}…
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

export default function ComparisonView({ llmConfig, isConfigured, availableCollections }) {
  const [leftTech, setLeftTech] = useState('')
  const [rightTech, setRightTech] = useState('')
  const [query, setQuery] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Find collection names that match selected techniques
  const leftCol = availableCollections?.find((c) => c.startsWith(leftTech + '_')) ?? null
  const rightCol = availableCollections?.find((c) => c.startsWith(rightTech + '_')) ?? null
  const canCompare = leftCol && rightCol && query.trim() && isConfigured

  async function handleCompare() {
    if (!canCompare) return
    setLoading(true)
    setError(null)
    try {
      const res = await compareDocuments({
        query: query.trim(),
        collection_names: [leftCol, rightCol],
        top_k: 3,
        llm_provider: llmConfig.provider,
        api_key: llmConfig.apiKey,
      })
      setResult(res.results)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function TechSelect({ value, onChange, label }) {
    return (
      <div className="flex flex-col gap-1">
        <span className="text-gray-500 text-xs">{label}</span>
        <select
          value={value}
          onChange={(e) => { onChange(e.target.value); setResult(null) }}
          className="bg-gray-800 text-gray-200 text-sm rounded-lg px-2 py-1.5 border border-gray-700 focus:outline-none focus:border-indigo-500 transition-all duration-200"
        >
          <option value="">— select technique —</option>
          {SLUGS.map((s) => <option key={s} value={s}>{s.replace(/_/g, ' ')}</option>)}
        </select>
      </div>
    )
  }

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 space-y-3">
      <div className="flex items-center gap-4 flex-wrap">
        <TechSelect value={leftTech} onChange={setLeftTech} label="Left technique" />
        <TechSelect value={rightTech} onChange={setRightTech} label="Right technique" />
        <div className="flex-1 min-w-48">
          <span className="text-gray-500 text-xs">Query</span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCompare()}
            placeholder="Ask both techniques the same question…"
            className="w-full bg-gray-800 text-gray-200 text-sm rounded-lg px-3 py-1.5 border border-gray-700 focus:outline-none focus:border-indigo-500 transition-all duration-200 mt-1"
          />
        </div>
        <button
          onClick={handleCompare}
          disabled={!canCompare || loading}
          className="mt-4 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200"
        >
          {loading ? 'Comparing…' : 'Compare'}
        </button>
      </div>

      {error && <p className="text-rose-400 text-xs">{error}</p>}
      {!isConfigured && <p className="text-amber-400 text-xs">⚠ Add an API key to enable comparison answers.</p>}

      <div className="flex gap-3">
        <ResultPanel label="Left"  collection={leftCol}  result={result?.[0]} />
        <ResultPanel label="Right" collection={rightCol} result={result?.[1]} />
      </div>

      {result && (
        <div className="flex justify-around bg-gray-800 rounded-lg p-2 text-xs text-gray-400">
          <span>Left latency: <strong className="text-indigo-400">{result[0].latency_ms} ms</strong></span>
          <span>Right latency: <strong className="text-emerald-400">{result[1].latency_ms} ms</strong></span>
          <span>Left chunks: <strong className="text-indigo-400">{result[0].retrieved_chunks?.length}</strong></span>
          <span>Right chunks: <strong className="text-emerald-400">{result[1].retrieved_chunks?.length}</strong></span>
        </div>
      )}
    </div>
  )
}

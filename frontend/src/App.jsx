import { useState, useEffect } from 'react'
import { getChunkers } from './utils/api'
import { useLLMConfig } from './hooks/useLLMConfig'
import APIKeyInput from './components/APIKeyInput'
import TechniqueSelector from './components/TechniqueSelector'
import DocumentPanel from './components/DocumentPanel'
import ChatInterface from './components/ChatInterface'
import MetricsPanel from './components/MetricsPanel'
import ComparisonView from './components/ComparisonView'

export default function App() {
  const [techniques, setTechniques] = useState([])
  const [selectedTechnique, setSelectedTechnique] = useState('')
  const [activeCollection, setActiveCollection] = useState(null)
  const [allIndexedCollections, setAllIndexedCollections] = useState([])
  const [indexStats, setIndexStats] = useState(null)
  const [compareMode, setCompareMode] = useState(false)
  const [error, setError] = useState(null)
  const llmConfig = useLLMConfig()

  useEffect(() => {
    getChunkers()
      .then(setTechniques)
      .catch((err) => setError(`Could not reach backend: ${err.message}`))
  }, [])

  function handleIndexed(collectionName, stats) {
    setActiveCollection(collectionName)
    setIndexStats({ ...stats, index_time_ms: null })
    setAllIndexedCollections((prev) =>
      prev.includes(collectionName) ? prev : [...prev, collectionName]
    )
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* ── Header ── */}
      <header className="bg-gray-950 border-b border-gray-800 px-6 py-3 flex items-center justify-between gap-4 sticky top-0 z-10">
        <div>
          <h1 className="text-indigo-400 font-bold text-xl leading-none">RAG Chunking Lab</h1>
          <p className="text-gray-500 text-xs mt-0.5">7 chunking techniques, side by side</p>
        </div>
        <div className="flex items-center gap-3">
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-500 hover:text-gray-300 text-sm transition-colors duration-200"
          >
            ★ GitHub
          </a>
          <APIKeyInput {...llmConfig} />
        </div>
      </header>

      {error && (
        <div className="bg-rose-900/30 border-b border-rose-800 px-6 py-2 text-rose-400 text-sm flex items-center justify-between">
          <span>{error}</span>
          <button
            onClick={() => setError(null)}
            className="text-rose-500 hover:text-rose-300 text-lg leading-none ml-4 transition-colors duration-200"
            aria-label="Dismiss"
          >
            ×
          </button>
        </div>
      )}

      {/* ── Technique selector ── */}
      <div className="px-6 py-4 border-b border-gray-800">
        <TechniqueSelector
          techniques={techniques}
          selected={selectedTechnique}
          onSelect={setSelectedTechnique}
          isConfigured={llmConfig.isConfigured}
        />
      </div>

      {/* ── Main panels ── */}
      <div className="flex flex-1 gap-4 px-6 py-4 min-h-0 overflow-hidden" style={{ height: 'calc(100vh - 220px)' }}>
        <div className="w-5/12 flex flex-col gap-3 min-h-0">
          <DocumentPanel
            selectedTechnique={selectedTechnique}
            llmConfig={llmConfig}
            onIndexed={handleIndexed}
          />
        </div>
        <div className="w-7/12 min-h-0">
          <ChatInterface
            collectionName={activeCollection}
            llmConfig={llmConfig}
            isConfigured={llmConfig.isConfigured}
          />
        </div>
      </div>

      {/* ── Metrics bar ── */}
      <MetricsPanel indexStats={indexStats} />

      {/* ── Compare mode ── */}
      <div className="px-6 py-3 border-t border-gray-800">
        <button
          onClick={() => setCompareMode((m) => !m)}
          className={`px-4 py-1.5 rounded-lg text-sm font-medium border transition-all duration-200 ${
            compareMode
              ? 'bg-indigo-500/20 border-indigo-500 text-indigo-300'
              : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-indigo-500/50'
          }`}
        >
          {compareMode ? '✕ Close Compare Mode' : '⇄ Compare Mode'}
        </button>
      </div>

      {compareMode && (
        <div className="px-6 pb-6">
          <ComparisonView
            llmConfig={llmConfig}
            isConfigured={llmConfig.isConfigured}
            availableCollections={allIndexedCollections}
          />
        </div>
      )}
    </div>
  )
}

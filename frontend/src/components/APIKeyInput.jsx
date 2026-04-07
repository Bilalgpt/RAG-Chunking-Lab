import { useState } from 'react'

const PROVIDERS = [
  { value: 'anthropic', label: 'Anthropic', color: 'bg-orange-400' },
  { value: 'openai',    label: 'OpenAI',    color: 'bg-green-400' },
  { value: 'groq',      label: 'Groq',      color: 'bg-purple-400' },
]

export default function APIKeyInput({ provider, apiKey, setProvider, setApiKey, isConfigured }) {
  const [show, setShow] = useState(false)

  return (
    <div className="flex items-center gap-3 bg-gray-900 border border-gray-700 rounded-xl px-4 py-2">
      {/* Provider dropdown */}
      <div className="flex items-center gap-2">
        <span className={`w-2 h-2 rounded-full ${PROVIDERS.find(p => p.value === provider)?.color}`} />
        <select
          value={provider}
          onChange={(e) => setProvider(e.target.value)}
          className="bg-gray-800 text-gray-200 text-sm rounded-lg px-2 py-1 border border-gray-700 focus:outline-none focus:border-indigo-500 transition-all duration-200"
        >
          {PROVIDERS.map((p) => (
            <option key={p.value} value={p.value}>{p.label}</option>
          ))}
        </select>
      </div>

      {/* API key input */}
      <div className="flex items-center gap-1 flex-1 max-w-xs">
        <input
          type={show ? 'text' : 'password'}
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="Paste API key…"
          className="bg-gray-800 text-gray-200 text-sm rounded-lg px-3 py-1 border border-gray-700 focus:outline-none focus:border-indigo-500 w-full transition-all duration-200"
        />
        <button
          onClick={() => setShow((s) => !s)}
          className="text-gray-400 hover:text-gray-200 text-xs px-1 transition-all duration-200"
          title={show ? 'Hide key' : 'Show key'}
        >
          {show ? '🙈' : '👁'}
        </button>
      </div>

      {/* Status */}
      {isConfigured ? (
        <span className="text-emerald-400 text-sm font-medium">✓ Ready</span>
      ) : (
        <span className="text-amber-400 text-xs">LLM features disabled — enter an API key</span>
      )}
    </div>
  )
}

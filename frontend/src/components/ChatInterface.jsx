import { useState, useEffect, useRef } from 'react'
import { queryDocument } from '../utils/api'

function scoreColor(score) {
  if (score > 0.7) return 'border-emerald-500'
  if (score > 0.4) return 'border-amber-500'
  return 'border-red-500'
}

function ChunkCard({ chunk }) {
  return (
    <div className={`border-l-2 ${scoreColor(chunk.score ?? 0)} bg-gray-900 rounded-r-lg p-2 mb-1`}>
      <div className="flex justify-between items-center mb-1">
        <span className="text-gray-500 text-xs">Chunk #{chunk.index}</span>
        <span className="text-xs font-mono text-gray-400">
          {chunk.score != null ? `${(chunk.score * 100).toFixed(1)}%` : ''}
        </span>
      </div>
      <p className="text-gray-300 text-xs font-mono leading-relaxed">
        {chunk.text?.slice(0, 150)}{chunk.text?.length > 150 ? '…' : ''}
      </p>
    </div>
  )
}

function AssistantMessage({ msg }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="flex flex-col gap-1 max-w-[85%]">
      <div className="bg-gray-800 rounded-xl px-4 py-3 text-gray-200 text-sm leading-relaxed">
        {msg.content}
      </div>
      <div className="flex items-center gap-3 px-1">
        <span className="text-gray-600 text-xs">{msg.latencyMs} ms</span>
        {msg.chunks?.length > 0 && (
          <button
            onClick={() => setOpen((o) => !o)}
            className="text-indigo-400 text-xs hover:text-indigo-300 transition-all duration-200"
          >
            {open ? '▲ Hide' : '▼ Show'} {msg.chunks.length} retrieved chunks
          </button>
        )}
      </div>
      {open && (
        <div className="mt-1 space-y-1 px-1">
          {msg.chunks.map((c, i) => <ChunkCard key={i} chunk={c} />)}
        </div>
      )}
    </div>
  )
}

export default function ChatInterface({ collectionName, llmConfig, isConfigured }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => { setMessages([]) }, [collectionName])
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const disabled = !collectionName

  async function handleSend() {
    if (!input.trim() || disabled || loading) return
    const question = input.trim()
    setInput('')
    setMessages((m) => [...m, { role: 'user', content: question }])
    setLoading(true)

    try {
      const result = await queryDocument({
        query: question,
        collection_name: collectionName,
        top_k: 5,
        llm_provider: llmConfig.provider,
        api_key: llmConfig.apiKey,
      })
      setMessages((m) => [...m, {
        role: 'assistant',
        content: result.answer,
        chunks: result.retrieved_chunks,
        latencyMs: result.latency_ms,
      }])
    } catch (err) {
      setMessages((m) => [...m, {
        role: 'assistant',
        content: `Error: ${err.message}`,
        chunks: [],
        latencyMs: null,
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full bg-gray-900 rounded-xl border border-gray-800">
      <div className="px-4 py-2.5 border-b border-gray-800 text-sm font-medium text-gray-400 flex items-center justify-between">
        <span>Chat {collectionName && <span className="text-indigo-400 text-xs ml-2 font-mono">{collectionName}</span>}</span>
        {messages.length > 0 && (
          <button
            onClick={() => setMessages([])}
            className="text-gray-600 hover:text-gray-400 text-xs transition-colors duration-200"
            title="Clear conversation"
          >
            Clear
          </button>
        )}
      </div>

      {/* Message thread */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4 min-h-0">
        {messages.length === 0 && (
          <p className="text-gray-600 text-sm text-center mt-8">
            {disabled ? 'Index a document first to start chatting.' : 'Ask anything about the indexed document.'}
          </p>
        )}
        {messages.map((msg, i) =>
          msg.role === 'user' ? (
            <div key={i} className="flex justify-end">
              <div className="bg-indigo-600 rounded-xl px-4 py-2 text-white text-sm max-w-[80%]">
                {msg.content}
              </div>
            </div>
          ) : (
            <div key={i} className="flex justify-start">
              <AssistantMessage msg={msg} />
            </div>
          )
        )}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-800 rounded-xl px-4 py-2 text-gray-400 text-sm animate-pulse">
              Thinking…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      {!isConfigured && collectionName && (
        <p className="text-amber-400 text-xs px-4 pb-1">
          No API key — chunks will be retrieved but answers are disabled.
        </p>
      )}
      <div className="flex gap-2 p-3 border-t border-gray-800">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
          disabled={disabled}
          placeholder={disabled ? 'Index a document first…' : 'Ask a question…'}
          className="flex-1 bg-gray-800 text-gray-200 text-sm rounded-lg px-3 py-2 border border-gray-700 focus:outline-none focus:border-indigo-500 disabled:opacity-40 transition-all duration-200"
        />
        <button
          onClick={handleSend}
          disabled={disabled || loading || !input.trim()}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200"
        >
          Send
        </button>
      </div>
    </div>
  )
}

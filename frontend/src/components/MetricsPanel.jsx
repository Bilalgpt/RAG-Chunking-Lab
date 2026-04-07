function Stat({ label, value }) {
  return (
    <div className="flex flex-col items-center px-5 py-2 border-r border-gray-800 last:border-0">
      <span className="text-gray-500 text-xs uppercase tracking-wide">{label}</span>
      <span className="text-gray-200 text-sm font-semibold mt-0.5">{value}</span>
    </div>
  )
}

export default function MetricsPanel({ indexStats }) {
  if (!indexStats) {
    return (
      <div className="bg-gray-900 border-t border-gray-800 px-4 py-2 text-gray-600 text-xs text-center">
        No document indexed yet
      </div>
    )
  }

  const { technique, chunk_count, avg_chunk_size, index_time_ms } = indexStats

  return (
    <div className="bg-gray-900 border-t border-gray-800 flex items-center">
      <Stat label="Technique"  value={technique.replace(/_/g, ' ')} />
      <Stat label="Chunks"     value={chunk_count} />
      <Stat label="Avg Size"   value={`${avg_chunk_size} words`} />
      <Stat label="Index Time" value={index_time_ms != null ? `${index_time_ms} ms` : '—'} />
    </div>
  )
}

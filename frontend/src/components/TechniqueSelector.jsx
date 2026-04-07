const COST = {
  fixed_size:   { level: 'Low',    color: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30', icon: '💚' },
  recursive:    { level: 'Low',    color: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30', icon: '💚' },
  semantic:     { level: 'Medium', color: 'bg-amber-500/20   text-amber-400   border-amber-500/30',   icon: '🟡' },
  hierarchical: { level: 'Low',    color: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30', icon: '💚' },
  late_chunking:{ level: 'Medium', color: 'bg-amber-500/20   text-amber-400   border-amber-500/30',   icon: '🟡' },
  contextual:   { level: 'High',   color: 'bg-rose-500/20    text-rose-400    border-rose-500/30',    icon: '🔴' },
  proposition:  { level: 'High',   color: 'bg-rose-500/20    text-rose-400    border-rose-500/30',    icon: '🔴' },
}

const BEST_FOR = {
  fixed_size:   'Uniform documents, logs, news',
  recursive:    'General purpose, mixed content',
  semantic:     'Multi-topic long documents',
  hierarchical: 'Technical docs with sections',
  late_chunking:'Narrative text, Wikipedia-style',
  contextual:   'Enterprise knowledge bases',
  proposition:  'Medical, legal, factual Q&A',
}

function SkeletonCard() {
  return (
    <div className="flex-none w-44 rounded-xl p-3 border border-gray-800 bg-gray-900 animate-pulse">
      <div className="h-3 bg-gray-700 rounded w-3/4 mb-2" />
      <div className="h-2 bg-gray-800 rounded w-full mb-1" />
      <div className="h-2 bg-gray-800 rounded w-5/6 mb-3" />
      <div className="h-4 bg-gray-700 rounded-full w-16 mb-2" />
      <div className="h-2 bg-gray-800 rounded w-4/5" />
    </div>
  )
}

export default function TechniqueSelector({ techniques, selected, onSelect, isConfigured }) {
  if (techniques.length === 0) {
    return (
      <div className="flex gap-3 overflow-x-auto pb-2 px-1">
        {Array.from({ length: 7 }).map((_, i) => <SkeletonCard key={i} />)}
      </div>
    )
  }

  return (
    <div className="flex gap-3 overflow-x-auto pb-2 px-1">
      {techniques.map((t) => {
        const cost = COST[t.slug] ?? COST.fixed_size
        const isSelected = selected === t.slug
        const needsKey = cost.level === 'High' && !isConfigured

        return (
          <button
            key={t.slug}
            onClick={() => onSelect(t.slug)}
            className={`flex-none w-44 text-left rounded-xl p-3 border transition-all duration-200 shadow-md
              ${isSelected
                ? 'border-indigo-500 bg-indigo-500/10 shadow-indigo-500/20'
                : 'border-gray-700 bg-gray-900 hover:border-indigo-500/60 hover:shadow-indigo-500/10'
              }`}
          >
            <div className="font-semibold text-gray-100 text-sm mb-1 capitalize">
              {t.slug.replace(/_/g, ' ')}
            </div>
            <div className="text-gray-400 text-xs mb-2 leading-snug line-clamp-2">{t.description}</div>
            <div className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full border ${cost.color} mb-1`}>
              {cost.icon} {cost.level}
            </div>
            <div className="text-gray-500 text-xs mt-1">
              <span className="text-gray-600">Best for: </span>{BEST_FOR[t.slug]}
            </div>
            {needsKey && (
              <div className="mt-1 text-amber-400 text-xs">⚠ Requires API key</div>
            )}
          </button>
        )
      })}
    </div>
  )
}

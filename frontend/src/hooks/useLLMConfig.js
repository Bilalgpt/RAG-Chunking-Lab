import { useState } from 'react'

export function useLLMConfig() {
  const [provider, setProvider] = useState('anthropic')
  const [apiKey, setApiKey] = useState('')

  return {
    provider,
    apiKey,
    setProvider,
    setApiKey,
    isConfigured: apiKey.length > 10,
  }
}

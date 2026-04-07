import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const client = axios.create({ baseURL: BASE_URL })

client.interceptors.response.use(
  (res) => res,
  (err) => {
    const detail = err.response?.data?.detail
    throw new Error(detail || err.response?.statusText || err.message)
  }
)

export const getChunkers = (signal) =>
  client.get('/api/chunkers', { signal }).then((r) => r.data)

export const indexDocument = (payload, signal) =>
  client.post('/api/documents/index', payload, { signal }).then((r) => r.data)

export const queryDocument = (payload, signal) =>
  client.post('/api/query', payload, { signal }).then((r) => r.data)

export const compareDocuments = (payload, signal) =>
  client.post('/api/compare', payload, { signal }).then((r) => r.data)

export const listCollections = (signal) =>
  client.get('/api/documents/list', { signal }).then((r) => r.data)

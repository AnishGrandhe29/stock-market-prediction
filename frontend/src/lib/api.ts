import axios from 'axios';
import { useAuthStore } from '@/stores/authStore';

const api = axios.create({
    baseURL: process.env.NEXT_PUBLIC_API_BASE_URL || '/api/v1',
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor to add auth token
api.interceptors.request.use((config) => {
    const token = useAuthStore.getState().accessToken;
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

// Response interceptor for token refresh
api.interceptors.response.use(
    (response) => response,
    async (error) => {
        const originalRequest = error.config;

        if (error.response?.status === 401 && !originalRequest._retry) {
            originalRequest._retry = true;

            const refreshToken = useAuthStore.getState().refreshToken;
            if (refreshToken) {
                try {
                    const response = await axios.post('/api/v1/auth/refresh', {
                        refresh_token: refreshToken,
                    });

                    const { access_token, refresh_token } = response.data;
                    useAuthStore.getState().login(access_token, refresh_token);

                    originalRequest.headers.Authorization = `Bearer ${access_token}`;
                    return api(originalRequest);
                } catch (refreshError) {
                    useAuthStore.getState().logout();
                    window.location.href = '/auth/login';
                }
            }
        }

        return Promise.reject(error);
    }
);

export default api;

// API functions
export const authAPI = {
    login: (email: string, password: string) =>
        api.post('/auth/login', { email, password }),
    register: (email: string, password: string, full_name?: string) =>
        api.post('/auth/register', { email, password, full_name }),
    googleLogin: () => window.location.href = '/api/v1/auth/google',
};

export const stocksAPI = {
    getRealtime: (symbol: string = '^NSEI') =>
        api.get(`/stocks/realtime/${symbol}`),
    getHistory: (symbol: string = '^NSEI', days: number = 60) =>
        api.get(`/stocks/history/${symbol}?days=${days}`),
    getTechnical: (symbol: string = '^NSEI', days: number = 60) =>
        api.get(`/stocks/technical/${symbol}?days=${days}`),
    getSentiment: (symbol: string = '^NSEI', days: number = 30) =>
        api.get(`/stocks/sentiment/${symbol}?days=${days}`),
    getMarketStatus: () => api.get('/stocks/market-status'),
};

export const predictionsAPI = {
    getLatest: (symbol: string = '^NSEI') =>
        api.get(`/predictions/latest?symbol=${symbol}`),
    getHistory: (symbol: string = '^NSEI', days: number = 30) =>
        api.get(`/predictions/history?symbol=${symbol}&days=${days}`),
    generate: (symbol: string = '^NSEI') =>
        api.post('/predictions/generate', { symbol }),
    getXAI: (predictionId: number) =>
        api.get(`/predictions/xai/${predictionId}`),
    getAccuracy: (period: string = 'weekly') =>
        api.get(`/predictions/accuracy?period=${period}`),
};

export const usersAPI = {
    getMe: () => api.get('/users/me'),
    updateProfile: (data: { full_name?: string }) =>
        api.patch('/users/me', data),
    getNotes: () => api.get('/users/notes'),
    createNote: (data: { title?: string; content: string; symbol?: string }) =>
        api.post('/users/notes', data),
    deleteNote: (id: number) => api.delete(`/users/notes/${id}`),
    getWatchlist: () => api.get('/users/watchlist'),
    addToWatchlist: (symbol: string) =>
        api.post('/users/watchlist', { symbol }),
    removeFromWatchlist: (id: number) =>
        api.delete(`/users/watchlist/${id}`),
    getAlerts: () => api.get('/users/alerts'),
    createAlert: (data: { symbol: string; alert_type: string; target_value: number }) =>
        api.post('/users/alerts', data),
    deleteAlert: (id: number) => api.delete(`/users/alerts/${id}`),
};

export const newsAPI = {
    getMarketNews: (watchlist?: string[]) => {
        const params = watchlist?.length ? `?watchlist=${watchlist.join(',')}` : '';
        return api.get(`/news/market${params}`);
    },
};


import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface User {
    id: number;
    email: string;
    full_name: string | null;
    avatar_url: string | null;
}

interface AuthState {
    user: User | null;
    accessToken: string | null;
    refreshToken: string | null;
    isAuthenticated: boolean;
    isLoading: boolean;
    login: (accessToken: string, refreshToken: string) => Promise<void>;
    logout: () => void;
    setUser: (user: User) => void;
}

export const useAuthStore = create<AuthState>()(
    persist(
        (set, get) => ({
            user: null,
            accessToken: null,
            refreshToken: null,
            isAuthenticated: false,
            isLoading: false,

            login: async (accessToken: string, refreshToken: string) => {
                set({ accessToken, refreshToken, isAuthenticated: true, isLoading: true });

                try {
                    const response = await fetch('/api/v1/users/me', {
                        headers: {
                            Authorization: `Bearer ${accessToken}`,
                        },
                    });

                    if (response.ok) {
                        const user = await response.json();
                        set({ user, isLoading: false });
                    } else {
                        set({ isLoading: false });
                    }
                } catch (error) {
                    console.error('Failed to fetch user:', error);
                    set({ isLoading: false });
                }
            },

            logout: () => {
                set({
                    user: null,
                    accessToken: null,
                    refreshToken: null,
                    isAuthenticated: false,
                });
            },

            setUser: (user: User) => {
                set({ user });
            },
        }),
        {
            name: 'auth-storage',
            partialize: (state) => ({
                accessToken: state.accessToken,
                refreshToken: state.refreshToken,
                isAuthenticated: state.isAuthenticated,
            }),
        }
    )
);

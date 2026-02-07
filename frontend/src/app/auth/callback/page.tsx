'use client';

import { useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useAuthStore } from '@/stores/authStore';

export default function AuthCallback() {
    const router = useRouter();
    const searchParams = useSearchParams();
    const { login } = useAuthStore();

    useEffect(() => {
        const accessToken = searchParams.get('access_token');
        const refreshToken = searchParams.get('refresh_token');

        if (accessToken && refreshToken) {
            login(accessToken, refreshToken).then(() => {
                router.push('/');
            });
        } else {
            router.push('/auth/login?error=oauth_failed');
        }
    }, [searchParams, login, router]);

    return (
        <div className= "min-h-screen flex items-center justify-center" >
        <div className="text-center" >
            <div className="w-12 h-12 border-4 border-primary-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                <p className="text-surface-600 dark:text-surface-400" >
                    Completing sign in...
    </p>
        </div>
        </div>
  );
}

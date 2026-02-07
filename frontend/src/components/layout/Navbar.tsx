'use client';

import Link from 'next/link';
import { useState } from 'react';
import {
    TrendingUp,
    Bell,
    User,
    Moon,
    Sun,
    LogOut,
    Settings,
    ChevronDown
} from 'lucide-react';
import { useAuthStore } from '@/stores/authStore';
import { InfoTooltip } from '@/components/ui/InfoTooltip';

export function Navbar() {
    const [isDark, setIsDark] = useState(false);
    const [showUserMenu, setShowUserMenu] = useState(false);
    const { user, isAuthenticated, logout } = useAuthStore();

    const toggleTheme = () => {
        const newTheme = !isDark;
        setIsDark(newTheme);
        document.documentElement.classList.toggle('dark', newTheme);
        localStorage.setItem('theme', newTheme ? 'dark' : 'light');
    };

    return (
        <nav className="fixed top-0 left-0 right-0 z-50 h-16 bg-white dark:bg-surface-800 border-b border-surface-200 dark:border-surface-700 px-6">
            <div className="flex items-center justify-between h-full">
                {/* Logo */}
                <Link href="/" className="flex items-center gap-2">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center">
                        <TrendingUp className="w-6 h-6 text-white" />
                    </div>
                    <div>
                        <h1 className="text-lg font-bold gradient-text">NIFTY 50 Predictor</h1>
                        <p className="text-xs text-surface-500">AI-Powered Predictions</p>
                    </div>
                </Link>

                {/* Right side */}
                <div className="flex items-center gap-4">
                    {/* Market Status */}
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface-100 dark:bg-surface-700">
                        <span className="live-indicator" />
                        <span className="ml-3 text-sm font-medium text-surface-600 dark:text-surface-300">
                            Market Open
                        </span>
                        <InfoTooltip
                            title="Market Status"
                            content="NSE trading hours: 9:15 AM - 3:30 PM IST, Monday to Friday"
                        />
                    </div>

                    {/* Notifications */}
                    <button className="relative p-2 rounded-lg hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors">
                        <Bell className="w-5 h-5 text-surface-600 dark:text-surface-300" />
                        <span className="absolute top-1 right-1 w-2 h-2 bg-danger-500 rounded-full" />
                    </button>

                    {/* Theme Toggle */}
                    <button
                        onClick={toggleTheme}
                        className="p-2 rounded-lg hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors"
                        aria-label="Toggle theme"
                    >
                        {isDark ? (
                            <Sun className="w-5 h-5 text-surface-600 dark:text-surface-300" />
                        ) : (
                            <Moon className="w-5 h-5 text-surface-600 dark:text-surface-300" />
                        )}
                    </button>

                    {/* User Menu */}
                    {isAuthenticated ? (
                        <div className="relative">
                            <button
                                onClick={() => setShowUserMenu(!showUserMenu)}
                                className="flex items-center gap-2 p-2 rounded-lg hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors"
                            >
                                <div className="w-8 h-8 rounded-full bg-primary-500 flex items-center justify-center text-white font-medium">
                                    {user?.email?.charAt(0).toUpperCase() || 'U'}
                                </div>
                                <ChevronDown className="w-4 h-4 text-surface-500" />
                            </button>

                            {showUserMenu && (
                                <div className="absolute right-0 mt-2 w-48 py-2 bg-white dark:bg-surface-800 rounded-xl shadow-lg border border-surface-200 dark:border-surface-700">
                                    <Link
                                        href="/settings"
                                        className="flex items-center gap-2 px-4 py-2 hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors"
                                    >
                                        <Settings className="w-4 h-4" />
                                        Settings
                                    </Link>
                                    <button
                                        onClick={logout}
                                        className="flex items-center gap-2 px-4 py-2 w-full text-left text-danger-500 hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors"
                                    >
                                        <LogOut className="w-4 h-4" />
                                        Logout
                                    </button>
                                </div>
                            )}
                        </div>
                    ) : (
                        <Link
                            href="/auth/login"
                            className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors font-medium"
                        >
                            Sign In
                        </Link>
                    )}
                </div>
            </div>
        </nav>
    );
}

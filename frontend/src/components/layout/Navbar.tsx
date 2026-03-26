'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import {
    TrendingUp, Bell, LogOut, Settings,
    ChevronDown, Zap, Sun, Moon
} from 'lucide-react';
import { useAuthStore } from '@/stores/authStore';
import { isNSEHoliday } from '@/lib/tradingDays';

function getMarketStatus(): { isOpen: boolean; message: string } {
    const now = new Date();
    const day = now.getDay();
    const mins = now.getHours() * 60 + now.getMinutes();
    if (day === 0 || day === 6) return { isOpen: false, message: day === 0 ? 'Closed (Sun)' : 'Closed (Sat)' };
    if (isNSEHoliday(now)) return { isOpen: false, message: 'Market Holiday' };
    if (mins < 555) return { isOpen: false, message: 'Pre-Market' };
    if (mins > 930) return { isOpen: false, message: 'After Hours' };
    return { isOpen: true, message: 'Market Open' };
}

export function Navbar() {
    const [isDark, setIsDark] = useState(true);
    const [showUserMenu, setShowUserMenu] = useState(false);
    const [marketStatus, setMarketStatus] = useState({ isOpen: false, message: '...' });
    const { user, isAuthenticated, logout } = useAuthStore();

    useEffect(() => {
        const update = () => setMarketStatus(getMarketStatus());
        update();
        const i = setInterval(update, 60000);
        return () => clearInterval(i);
    }, []);

    useEffect(() => {
        if (typeof window !== 'undefined') {
            document.documentElement.classList.add('dark');
            localStorage.setItem('theme', 'dark');
            setIsDark(true);
        }
    }, []);

    const toggleTheme = () => {
        const next = !isDark;
        setIsDark(next);
        document.documentElement.classList.toggle('dark', next);
        localStorage.setItem('theme', next ? 'dark' : 'light');
    };

    return (
        <nav
            className="fixed top-0 left-0 right-0 z-50 h-16 glass"
            style={{ borderBottom: '1px solid var(--border-ghost)' }}
        >
            <div className="flex items-center justify-between h-full px-6">

                {/* ── Logo ── */}
                <Link href="/" className="flex items-center gap-3 min-w-[200px]">
                    <div
                        className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
                        style={{
                            background: 'linear-gradient(135deg, var(--color-primary-dim) 0%, var(--color-primary-container) 100%)',
                            boxShadow: '0 0 18px rgba(192,193,255,0.35), inset 0 1px 0 rgba(255,255,255,0.2)',
                        }}
                    >
                        <TrendingUp className="w-5 h-5 text-white" />
                    </div>
                    <div>
                        <p className="text-base font-bold gradient-text leading-tight">NIFTY AI</p>
                        <p className="label-upper" style={{ fontSize: '0.58rem' }}>Prediction System</p>
                    </div>
                </Link>

                {/* ── Center: GIFT NIFTY Signal ── */}
                <div
                    className="flex items-center gap-2 px-4 py-2 rounded-xl"
                    style={{
                        background: 'rgba(245,158,11,0.10)',
                        border: '1px solid rgba(245,158,11,0.22)',
                        boxShadow: '0 0 14px rgba(245,158,11,0.10)',
                    }}
                >
                    <Zap className="w-4 h-4" style={{ color: 'var(--color-amber)' }} />
                    <span className="text-sm font-bold" style={{ color: 'var(--color-amber)' }}>
                        GIFT NIFTY
                    </span>
                    <span className="badge badge-amber">Signal Active</span>
                </div>

                {/* ── Right Controls ── */}
                <div className="flex items-center gap-2">

                    {/* Market Status Pill */}
                    <div
                        className="flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-semibold"
                        style={{
                            background: marketStatus.isOpen
                                ? 'rgba(78,222,163,0.10)'
                                : 'rgba(145,143,154,0.10)',
                            color: marketStatus.isOpen ? 'var(--color-emerald)' : 'var(--text-muted)',
                            border: `1px solid ${marketStatus.isOpen
                                ? 'rgba(78,222,163,0.25)'
                                : 'rgba(145,143,154,0.18)'}`,
                        }}
                    >
                        {marketStatus.isOpen
                            ? <span className="pulse-green" />
                            : <span className="w-2 h-2 rounded-full inline-block" style={{ background: 'var(--outline-color)' }} />
                        }
                        <span style={{ fontSize: '0.8rem' }}>{marketStatus.message}</span>
                    </div>

                    {/* Notifications */}
                    <button
                        className="relative p-2 rounded-lg btn-ghost"
                        style={{ border: 'none', padding: '8px' }}
                        aria-label="Notifications"
                    >
                        <Bell className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
                        <span
                            className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full"
                            style={{ background: 'var(--color-rose)' }}
                        />
                    </button>

                    {/* Theme Toggle */}
                    <button
                        onClick={toggleTheme}
                        className="p-2 rounded-lg btn-ghost"
                        style={{ border: 'none', padding: '8px' }}
                        aria-label="Toggle theme"
                    >
                        {isDark
                            ? <Sun className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
                            : <Moon className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
                        }
                    </button>

                    {/* User Menu */}
                    {isAuthenticated ? (
                        <div className="relative">
                            <button
                                onClick={() => setShowUserMenu(!showUserMenu)}
                                className="flex items-center gap-2 px-3 py-1.5 rounded-lg btn-ghost"
                                style={{ border: 'none' }}
                            >
                                {user?.avatar_url ? (
                                    <img src={user.avatar_url} alt="Profile"
                                        className="w-7 h-7 rounded-full object-cover" />
                                ) : (
                                    <div
                                        className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold"
                                        style={{
                                            background: 'linear-gradient(135deg, var(--color-primary-dim), var(--color-primary-container))',
                                            color: 'var(--color-on-primary)',
                                        }}
                                    >
                                        {user?.full_name?.charAt(0).toUpperCase()
                                            || user?.email?.charAt(0).toUpperCase()
                                            || 'U'}
                                    </div>
                                )}
                                <ChevronDown className="w-3 h-3" style={{ color: 'var(--text-muted)' }} />
                            </button>

                            {showUserMenu && (
                                <div
                                    className="absolute right-0 mt-2 w-52 py-2 rounded-xl glass-ai"
                                    style={{ boxShadow: '0 20px 60px rgba(0,0,0,0.5)' }}
                                >
                                    <div className="px-4 py-2 mb-1" style={{ borderBottom: '1px solid var(--border-ghost)' }}>
                                        <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                                            {user?.full_name || 'User'}
                                        </p>
                                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{user?.email}</p>
                                    </div>
                                    <Link href="/settings"
                                        className="flex items-center gap-2 px-4 py-2 text-sm nav-item rounded-none"
                                        onClick={() => setShowUserMenu(false)}
                                    >
                                        <Settings className="w-4 h-4" /> Settings
                                    </Link>
                                    <button
                                        onClick={() => { logout(); setShowUserMenu(false); }}
                                        className="flex items-center gap-2 px-4 py-2 w-full text-left text-sm transition-colors"
                                        style={{ color: 'var(--color-rose)' }}
                                    >
                                        <LogOut className="w-4 h-4" /> Logout
                                    </button>
                                </div>
                            )}
                        </div>
                    ) : (
                        <Link href="/auth/login" className="btn-primary text-sm">
                            Sign In
                        </Link>
                    )}
                </div>
            </div>
        </nav>
    );
}

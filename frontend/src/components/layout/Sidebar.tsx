'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    LayoutDashboard,
    LineChart,
    Brain,
    BookOpen,
    Bell,
    Star,
    FileText,
    HelpCircle,
    Cpu,
    TrendingUp,
} from 'lucide-react';
import clsx from 'clsx';

const menuItems = [
    {
        label: 'Dashboard',
        href: '/',
        icon: LayoutDashboard,
    },
    {
        label: 'Live Chart',
        href: '/chart',
        icon: LineChart,
    },
    {
        label: 'Predictions',
        href: '/predictions',
        icon: Brain,
    },
    {
        label: 'XAI Insights',
        href: '/xai',
        icon: HelpCircle,
    },
    {
        label: 'Model Info',
        href: '/model',
        icon: Cpu,
    },
    { divider: true },
    {
        label: 'Watchlist',
        href: '/watchlist',
        icon: Star,
    },
    {
        label: 'Notes',
        href: '/notes',
        icon: FileText,
    },
    {
        label: 'Alerts',
        href: '/alerts',
        icon: Bell,
    },
    { divider: true },
    {
        label: 'Docs',
        href: '/docs',
        icon: BookOpen,
    },
];

export function Sidebar() {
    const pathname = usePathname();

    return (
        <aside
            className="fixed left-0 top-16 bottom-0 w-60 overflow-y-auto flex flex-col"
            style={{
                background: 'var(--surface-low)',
                borderRight: '1px solid var(--border-ghost)',
            }}
        >
            {/* Branding accent */}
            <div className="px-4 pt-5 pb-3">
                <div
                    className="flex items-center gap-2 px-3 py-2 rounded-xl"
                    style={{
                        background: 'rgba(192,193,255,0.07)',
                        border: '1px solid rgba(192,193,255,0.14)',
                    }}
                >
                    <TrendingUp className="w-4 h-4" style={{ color: 'var(--color-primary)' }} />
                    <div>
                        <p className="text-xs font-bold gradient-text">NIFTY 50 AI</p>
                        <p className="label-upper" style={{ fontSize: '0.55rem' }}>Prediction System</p>
                    </div>
                    {/* Live pulse dot */}
                    <span className="ml-auto pulse-indigo" />
                </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 px-3 pb-6 space-y-0.5">
                {menuItems.map((item, index) => {
                    if ('divider' in item) {
                        return (
                            <div
                                key={index}
                                className="my-3 mx-2"
                                style={{ height: '1px', background: 'var(--border-faint)' }}
                            />
                        );
                    }

                    const Icon = item.icon;
                    const isActive = pathname === item.href;

                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={clsx(
                                'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 group',
                                isActive ? 'nav-item-active' : 'nav-item'
                            )}
                        >
                            <Icon
                                className="w-4 h-4 flex-shrink-0 transition-colors"
                                style={{
                                    color: isActive ? 'var(--color-primary)' : undefined,
                                }}
                            />
                            <span className="text-sm font-medium">{item.label}</span>
                            {isActive && (
                                <span
                                    className="ml-auto w-1.5 h-1.5 rounded-full"
                                    style={{ background: 'var(--color-primary)' }}
                                />
                            )}
                        </Link>
                    );
                })}
            </nav>

            {/* Footer */}
            <div
                className="px-4 py-4"
                style={{ borderTop: '1px solid var(--border-ghost)' }}
            >
                <p className="label-upper text-center">ACMI++ v2.1.0 · Live</p>
            </div>
        </aside>
    );
}

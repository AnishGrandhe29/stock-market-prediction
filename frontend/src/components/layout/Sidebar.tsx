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
    HelpCircle
} from 'lucide-react';
import { InfoTooltip } from '@/components/ui/InfoTooltip';
import clsx from 'clsx';

const menuItems = [
    {
        label: 'Dashboard',
        href: '/',
        icon: LayoutDashboard,
        tooltip: 'Overview of NIFTY 50 with latest predictions and market status',
    },
    {
        label: 'Live Chart',
        href: '/chart',
        icon: LineChart,
        tooltip: 'Real-time candlestick chart with technical indicators',
    },
    {
        label: 'Predictions',
        href: '/predictions',
        icon: Brain,
        tooltip: 'AI predictions with confidence levels and historical accuracy',
    },
    {
        label: 'XAI Insights',
        href: '/xai',
        icon: HelpCircle,
        tooltip: 'Explainable AI - understand why the model made its predictions',
    },
    { divider: true },
    {
        label: 'Watchlist',
        href: '/watchlist',
        icon: Star,
        tooltip: 'Track your favorite stocks and set price alerts',
    },
    {
        label: 'Notes',
        href: '/notes',
        icon: FileText,
        tooltip: 'Personal notes and trading journal',
    },
    {
        label: 'Alerts',
        href: '/alerts',
        icon: Bell,
        tooltip: 'Configure price and prediction alerts',
    },
    { divider: true },
    {
        label: 'Documentation',
        href: '/docs',
        icon: BookOpen,
        tooltip: 'Learn about the model, features, and how to use the platform',
    },
];

export function Sidebar() {
    const pathname = usePathname();

    return (
        <aside className= "fixed left-0 top-16 bottom-0 w-64 bg-white dark:bg-surface-800 border-r border-surface-200 dark:border-surface-700 p-4 overflow-y-auto" >
        <nav className="space-y-1" >
        {
            menuItems.map((item, index) => {
                if ('divider' in item) {
                    return <hr key={ index } className = "my-4 border-surface-200 dark:border-surface-700" />;
                }

                const Icon = item.icon;
                const isActive = pathname === item.href;

                return (
                    <Link
              key= { item.href }
                href = { item.href }
                className = {
                    clsx(
                'flex items-center justify-between px-4 py-3 rounded-xl transition-all duration-200',
                        isActive
                            ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 font-medium'
                  : 'text-surface-600 dark:text-surface-400 hover:bg-surface-100 dark:hover:bg-surface-700'
                    )
                }
                    >
                    <div className="flex items-center gap-3" >
                        <Icon className="w-5 h-5" />
                            <span>{ item.label } </span>
                            </div>
                            < InfoTooltip title = { item.label } content = { item.tooltip } />
                                </Link>
          );
        })
}
</nav>

{/* Model Info Card */ }
<div className="mt-6 p-4 rounded-xl bg-gradient-to-br from-primary-500/10 to-primary-700/10 border border-primary-200 dark:border-primary-800" >
    <div className="flex items-center gap-2 mb-2" >
        <Brain className="w-5 h-5 text-primary-500" />
            <span className="font-medium text-primary-700 dark:text-primary-300" > Model Status </span>
                </div>
                < p className = "text-sm text-surface-600 dark:text-surface-400" >
                    Multimodal TCN - BERT
                        </p>
                        < div className = "mt-2 flex items-center gap-2" >
                            <span className="w-2 h-2 bg-success-500 rounded-full" />
                                <span className="text-xs text-success-600 dark:text-success-400" > Online </span>
                                    </div>
                                    </div>
                                    </aside>
  );
}

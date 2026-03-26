import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Providers } from '@/components/Providers';
import { Navbar } from '@/components/layout/Navbar';
import { Sidebar } from '@/components/layout/Sidebar';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });

export const metadata: Metadata = {
    title: 'NIFTY AI — ACMI++ Prediction System',
    description: 'ACMI++ multimodal deep learning predictions for NIFTY 50 with Explainable AI and GIFT NIFTY integration',
    keywords: ['NIFTY 50', 'GIFT NIFTY', 'stock prediction', 'AI', 'ACMI++', 'XAI'],
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" suppressHydrationWarning>
            <body className={inter.className} style={{ background: 'var(--surface-base)' }}>
                <Providers>
                    <div className="min-h-screen" style={{ background: 'var(--surface-base)' }}>
                        <Navbar />
                        <div className="flex">
                            <Sidebar />
                            <main
                                className="flex-1 ml-60 pt-16 min-h-screen"
                                style={{ background: 'var(--surface-base)' }}
                            >
                                <div className="p-6 max-w-screen-2xl mx-auto">
                                    {children}
                                </div>
                            </main>
                        </div>
                    </div>
                </Providers>
            </body>
        </html>
    );
}

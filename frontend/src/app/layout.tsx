import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Providers } from '@/components/Providers';
import { Navbar } from '@/components/layout/Navbar';
import { Sidebar } from '@/components/layout/Sidebar';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
    title: 'NIFTY 50 Index Predictor - AI-Powered Market Predictions',
    description: 'Multimodal deep learning predictions for NIFTY 50 Index with Explainable AI',
    keywords: ['NIFTY 50', 'stock prediction', 'AI', 'machine learning', 'XAI'],
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang= "en" suppressHydrationWarning >
            <body className={ inter.className }>
                <Providers>
                <div className="min-h-screen bg-surface-50 dark:bg-surface-900" >
                    <Navbar />
                    < div className = "flex" >
                        <Sidebar />
                        < main className = "flex-1 ml-64 pt-16 p-6" >
                            { children }
                            </main>
                            </div>
                            </div>
                            </Providers>
                            </body>
                            </html>
  );
}

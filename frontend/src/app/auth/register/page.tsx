'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Mail, Lock, User, Eye, EyeOff, AlertCircle, CheckCircle } from 'lucide-react';
import { authAPI } from '@/lib/api';
import { InfoTooltip } from '@/components/ui/InfoTooltip';

export default function RegisterPage() {
    const router = useRouter();

    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [fullName, setFullName] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const passwordRequirements = [
        { label: 'At least 8 characters', met: password.length >= 8 },
        { label: 'Contains a number', met: /\d/.test(password) },
        { label: 'Contains uppercase letter', met: /[A-Z]/.test(password) },
    ];

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');

        if (password.length < 8) {
            setError('Password must be at least 8 characters');
            return;
        }

        setIsLoading(true);

        try {
            await authAPI.register(email, password, fullName);
            router.push('/auth/login?registered=true');
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Registration failed. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className= "min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-50 to-primary-100 dark:from-surface-900 dark:to-surface-800 p-4" >
        <div className="w-full max-w-md" >
            {/* Logo */ }
            < div className = "text-center mb-8" >
                <div className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center mb-4" >
                    <span className="text-3xl font-bold text-white" > N </span>
                        </div>
                        < h1 className = "text-2xl font-bold text-surface-900 dark:text-white" >
                            Create Account
                                </h1>
                                < p className = "text-surface-500 mt-1" >
                                    Start predicting NIFTY 50 with AI
                                    </p>
                                    </div>

        {/* Register Card */ }
    <div className="card p-8" >
        <form onSubmit={ handleSubmit } className = "space-y-5" >
            {/* Error */ }
    {
        error && (
            <div className="flex items-center gap-2 p-3 bg-danger-50 dark:bg-danger-900/30 border border-danger-200 dark:border-danger-800 rounded-lg text-danger-600 dark:text-danger-400" >
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                    <span className="text-sm" > { error } </span>
                        </div>
            )
    }

    {/* Full Name */ }
    <div>
        <div className="flex items-center justify-between mb-2" >
            <label className="text-sm font-medium text-surface-700 dark:text-surface-300" >
                Full Name
                    </label>
                    < InfoTooltip title = "Name" content = "Your display name on the platform." />
                        </div>
                        < div className = "relative" >
                            <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-surface-400" />
                                <input
                  type="text"
    value = { fullName }
    onChange = {(e) => setFullName(e.target.value)
}
placeholder = "John Doe"
className = "w-full pl-10 pr-4 py-3 rounded-lg border border-surface-200 dark:border-surface-600 bg-white dark:bg-surface-700 text-surface-900 dark:text-white placeholder-surface-400 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
    />
    </div>
    </div>

{/* Email */ }
<div>
    <div className="flex items-center justify-between mb-2" >
        <label className="text-sm font-medium text-surface-700 dark:text-surface-300" >
            Email Address
                </label>
                </div>
                < div className = "relative" >
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-surface-400" />
                        <input
                  type="email"
value = { email }
onChange = {(e) => setEmail(e.target.value)}
required
placeholder = "you@example.com"
className = "w-full pl-10 pr-4 py-3 rounded-lg border border-surface-200 dark:border-surface-600 bg-white dark:bg-surface-700 text-surface-900 dark:text-white placeholder-surface-400 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
    />
    </div>
    </div>

{/* Password */ }
<div>
    <div className="flex items-center justify-between mb-2" >
        <label className="text-sm font-medium text-surface-700 dark:text-surface-300" >
            Password
            </label>
            </div>
            < div className = "relative" >
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-surface-400" />
                    <input
                  type={ showPassword ? 'text' : 'password' }
value = { password }
onChange = {(e) => setPassword(e.target.value)}
required
placeholder = "••••••••"
className = "w-full pl-10 pr-12 py-3 rounded-lg border border-surface-200 dark:border-surface-600 bg-white dark:bg-surface-700 text-surface-900 dark:text-white placeholder-surface-400 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
    />
    <button
                  type="button"
onClick = {() => setShowPassword(!showPassword)}
className = "absolute right-3 top-1/2 -translate-y-1/2 text-surface-400 hover:text-surface-600"
    >
    { showPassword?<EyeOff className = "w-5 h-5" /> : <Eye className="w-5 h-5" />}
</button>
    </div>

{/* Password Requirements */ }
<div className="mt-3 space-y-1.5" >
    {
        passwordRequirements.map((req) => (
            <div key= { req.label } className = "flex items-center gap-2" >
            <CheckCircle
                      className={`w-4 h-4 ${req.met ? 'text-success-500' : 'text-surface-300'
            }`}
    />
    <span
                      className={
    `text-xs ${req.met ? 'text-success-600' : 'text-surface-400'
    }`
}
                    >
    { req.label }
    </span>
    </div>
                ))}
</div>
    </div>

{/* Submit */ }
<button
              type="submit"
disabled = { isLoading }
className = "w-full py-3 bg-gradient-to-r from-primary-500 to-primary-600 text-white font-semibold rounded-lg hover:from-primary-600 hover:to-primary-700 focus:ring-4 focus:ring-primary-500/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
    >
    { isLoading? 'Creating account...': 'Create Account' }
    </button>
    </form>

{/* Login Link */ }
<p className="mt-6 text-center text-sm text-surface-500" >
    Already have an account ? { ' '}
        < Link href = "/auth/login" className = "text-primary-500 hover:text-primary-600 font-medium" >
            Sign in
            </Link>
            </p>
            </div>
            </div>
            </div>
  );
}

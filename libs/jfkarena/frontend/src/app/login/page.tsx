import { LoginForm } from '@/components/login-form'
import { Suspense } from 'react'

export default function LoginPage() {
  return (
    <div className="flex min-h-svh flex-col items-center gap-6 bg-background p-6 md:p-10">
      <div className="w-full max-w-sm mt-40">
        <Suspense>
          <LoginForm />
        </Suspense>
      </div>
    </div>
  )
}

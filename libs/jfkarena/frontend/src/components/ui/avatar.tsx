'use client'

import { cn } from '@/lib/utils'
import * as AvatarPrimitive from '@radix-ui/react-avatar'

import * as React from 'react'

function Avatar({ ref, className, ...props }: React.ComponentPropsWithoutRef<typeof AvatarPrimitive.Root> & { ref?: React.RefObject<React.ElementRef<typeof AvatarPrimitive.Root> | null> }) {
  return (
    <AvatarPrimitive.Root
      ref={ref}
      className={cn(
        'relative flex h-10 w-10 shrink-0 overflow-hidden rounded-full',
        className,
      )}
      {...props}
    />
  )
}
Avatar.displayName = AvatarPrimitive.Root.displayName

function AvatarImage({ ref, className, ...props }: React.ComponentPropsWithoutRef<typeof AvatarPrimitive.Image> & { ref?: React.RefObject<React.ElementRef<typeof AvatarPrimitive.Image> | null> }) {
  return (
    <AvatarPrimitive.Image
      ref={ref}
      className={cn('aspect-square h-full w-full', className)}
      {...props}
    />
  )
}
AvatarImage.displayName = AvatarPrimitive.Image.displayName

function AvatarFallback({ ref, className, ...props }: React.ComponentPropsWithoutRef<typeof AvatarPrimitive.Fallback> & { ref?: React.RefObject<React.ElementRef<typeof AvatarPrimitive.Fallback> | null> }) {
  return (
    <AvatarPrimitive.Fallback
      ref={ref}
      className={cn(
        'flex h-full w-full items-center justify-center rounded-full bg-muted',
        className,
      )}
      {...props}
    />
  )
}
AvatarFallback.displayName = AvatarPrimitive.Fallback.displayName

export { Avatar, AvatarFallback, AvatarImage }

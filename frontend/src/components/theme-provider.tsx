'use client'

import * as React from 'react'

type Theme = 'light' | 'dark' | 'system'

interface ThemeProviderProps {
  children: React.ReactNode
  attribute?: string
  defaultTheme?: Theme
  enableSystem?: boolean
  disableTransitionOnChange?: boolean
}

const ThemeContext = React.createContext<{ theme: Theme; setTheme: (theme: Theme) => void }>({
  theme: 'light',
  setTheme: () => {},
})

export function ThemeProvider({
  children,
  attribute = 'class',
  defaultTheme = 'system',
  enableSystem = true,
  disableTransitionOnChange = false,
}: ThemeProviderProps) {
  const [theme, setTheme] = React.useState<Theme>(() => {
    if (typeof window === 'undefined') return defaultTheme
    const stored = window.localStorage.getItem('theme')
    if (stored === 'light' || stored === 'dark') return stored
    return defaultTheme
  })

  React.useEffect(() => {
    if (!enableSystem && theme === 'system') {
      setTheme('light')
      return
    }
    let appliedTheme = theme
    if (theme === 'system' && enableSystem) {
      const mql = window.matchMedia('(prefers-color-scheme: dark)')
      appliedTheme = mql.matches ? 'dark' : 'light'
    }
    if (attribute === 'class') {
      document.documentElement.classList.remove('light', 'dark')
      document.documentElement.classList.add(appliedTheme)
    } else {
      document.documentElement.setAttribute(attribute, appliedTheme)
    }
    window.localStorage.setItem('theme', theme)
    // Optionally disable transition on theme change
    if (disableTransitionOnChange) {
      const style = document.createElement('style')
      style.innerHTML = '* { transition: none !important; }'
      document.head.appendChild(style)
      setTimeout(() => {
        document.head.removeChild(style)
      }, 0)
    }
  }, [theme, attribute, enableSystem, disableTransitionOnChange])

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  return React.useContext(ThemeContext)
}

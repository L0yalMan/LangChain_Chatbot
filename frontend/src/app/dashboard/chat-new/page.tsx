"use client"

import type React from "react"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Send } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import axios from "axios"
import { supabase } from "@/lib/supabase"

type Message = {
  id: string
  content: string
  role: "user" | "ai"
  created_at: Date
}

type User = {
  id: string
}


export default function ChatInterface() {
  const router = useRouter()
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState("")
  const [user, setUser] = useState<User | null>(null)
  const [isAiLoading, setIsAiLoading] = useState(false)
  useEffect(() => {
    const getUser = async () => {
      const { data: { user } } = await supabase.auth.getUser()
      if(user) {
        setUser(user as User)
      }
    }
    
    getUser()
  }, [])

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return

    const newMessage: Message = {
      id: (Date.now() + 1).toString(),
      content: inputValue,
      role: "user",
      created_at: new Date(),
    }

    setMessages((prev) => [...prev, newMessage])

    setTimeout(async () => {
      setIsAiLoading(true)
      const response = await axios.post(`https://fa2c5e5b19fc.ngrok-free.app/chat`, {
        question: inputValue,
        chat_history: [],
        headers: {
          "ngrok-skip-browser-warning": "true" // Bypass ngrok warning
        }
      })
      const answer = response.data.answer

      const response2 = await axios.post(`https://fa2c5e5b19fc.ngrok-free.app/chat`, {
        question: answer + 'Please give me one sentences which express above things and it must be less than 15 letters',
        chat_history: [
          {
            role: 'user',
            content: inputValue
          },
          {
            role: 'ai',
            content: answer
          }
        ],
        headers: {
          "ngrok-skip-browser-warning": "true" // Bypass ngrok warning
        }
      })
      const chatTitle = response2.data.answer

      const { data: chatSessionData, error: chatSessionError } = await supabase.from('chat_sessions').insert({
        chat_title: chatTitle,
        user_id: user?.id,
      }).select().single()

      const { data: chatHistoryData, error: chatHistoryError } = await supabase.from('chat_history').insert({
        user_id: user?.id,
        session_id: chatSessionData?.id,
        content: inputValue,
        role: "user",
        created_at: new Date().toISOString(),
      }).select().single()

      const { data, error } = await supabase.from('chat_history').insert({
        user_id: user?.id,
        session_id: chatSessionData?.id,
        content: answer,
        role: "ai",
        created_at: new Date().toISOString(),
      }).select().single()
      
      router.push(`/dashboard/chat/${chatSessionData?.id}`)
    }, 1000)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  if (messages.length === 0) {
    return (
      <div className="flex flex-col h-full">
        {/* New Chat Header */}
        <div className="border-b border-gray-200 p-4">
          <h2 className="text-lg font-semibold text-gray-900">
            New Chat
          </h2>
          <p className="text-sm text-gray-600 mt-1">
            Start a new conversation
          </p>
        </div>

        {/* Empty State */}
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="text-gray-400 mb-4">
              <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">Start a new conversation</h3>
            <p className="text-gray-600 mb-4">Type your message below to begin chatting</p>
          </div>
        </div>

        {/* Input */}
        { messages.length === 0 && (
          <div className="border-t border-gray-200 p-4">
            <div className="flex space-x-2">
              <Input
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message..."
                className="flex-1"
              />
              <Button onClick={handleSendMessage} disabled={!inputValue.trim()}>
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Chat Header */}
      <div className="border-b border-gray-200 p-4">
        <h2 className="text-lg font-semibold text-gray-900">
          New Chat
        </h2>
      </div>

        {/* Messages */}
        <ScrollArea className="flex-1 p-4">
        <div className="space-y-4">
        {messages.map((message) => (
            <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
                className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                message.role === "user" ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-900"
                }`}
            >
                <p className="text-sm">{message.content}</p>
                <p className="text-xs mt-1 opacity-70">
                {new Date(message.created_at).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                })}
                </p>
            </div>
            </div>
        ))}
        
        {/* Loading indicator */}
        {isAiLoading && (
        <div className="flex justify-start">
            <div className="bg-gray-200 text-gray-900 max-w-xs lg:max-w-md px-4 py-2 rounded-lg">
            <div className="flex items-center space-x-2">
                <div className="flex space-x-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
                <span className="text-sm text-gray-600">AI is thinking...</span>
            </div>
            </div>
        </div>
        )}
        </div>
        </ScrollArea>


      {/* Input */}
      <div className="border-t border-gray-200 p-4">
        <div className="flex space-x-2">
          <Input
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            className="flex-1"
          />
          <Button onClick={handleSendMessage} disabled={!inputValue.trim()}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}

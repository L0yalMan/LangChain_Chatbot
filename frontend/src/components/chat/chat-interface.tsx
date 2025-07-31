"use client"

import React, { useEffect, useState } from "react"
import { Send } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import axios from "axios"
import { supabase } from "@/lib/supabase"
import ReactMarkdown from "react-markdown"

type Message = {
  id: string
  content: string
  role: "user" | "ai"
  created_at: Date
}

type User = {
  id: string
}


export default function ChatInterface({ chatId }: { chatId: string }) {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState("")
  const [user, setUser] = useState<User | null>(null)
  const [isAiLoading, setIsAiLoading] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [chatTitle, setChatTitle] = useState("")
  useEffect(() => {
    const getUser = async () => {
      setIsLoading(true)
      const { data: { user } } = await supabase.auth.getUser()
      if(user) {
        const { data: chatSessionData, error: chatSessionError } = await supabase.from('chat_sessions').select('chat_title').eq('id', chatId).single()
        if(chatSessionData) {
          setChatTitle(chatSessionData.chat_title)
        }
        const {data: chatHistoryData, error: chatHistoryError} = await supabase.from('chat_history').select('*').eq('user_id', user.id)
        .eq('session_id', chatId).order('created_at', { ascending: true })
        if(chatHistoryData) {
          setMessages(chatHistoryData as Message[])
        }
        setUser(user as User)
      }
      setIsLoading(false)
    }
    
    getUser()
  }, [])

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return

    const { data, error } = await supabase.from('chat_history').insert({
      user_id: user?.id,
      session_id: chatId,
      content: inputValue,
      role: "user",
      created_at: new Date().toISOString(),
    }).select().single()

    if (error) console.error('Error sending message:', error)
    else console.log('Message sent:', data)

    const newMessage: Message = {
      id: data.id,
      content: inputValue,
      role: "user",
      created_at: new Date(),
    }

    setMessages((prev) => [...prev, newMessage])
    setInputValue("")

    setTimeout(async () => {
      setIsAiLoading(true)
      const chat_history = messages.map(message => {
        return {
          role: message.role,
          content: message.content
        }
      })
      console.log('chat_history--------->>>>>>>>>>>>>', chat_history);
      const response = await axios.post(`${process.env.NEXT_PUBLIC_BACKEND_HOST}/chat`, {
        question: inputValue, chat_history, 
          headers: {
          "ngrok-skip-browser-warning": "true" // Bypass ngrok warning
        }
       })
      const answer = response.data.answer
      
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: answer,
        role: "ai",
        created_at: new Date(),
      }
      setMessages((prev) => [...prev, aiResponse])

      const { data, error } = await supabase.from('chat_history').insert({
        user_id: user?.id,
        session_id: chatId,
        content: answer,
        role: "ai",
        created_at: new Date().toISOString(),
      }).select().single()
  
      if (error) console.error('Error sending message:', error)
      else console.log('Message sent:', data)
      setIsAiLoading(false)
    }, 1000)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  if (chatId === "") {
    return (
      <div className="flex flex-col h-full">  
        {/* Empty State */}
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="text-gray-400 mb-4">
              <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">Start a new conversation</h3>
            <p className="text-gray-600 mb-4">Click New Chat Button</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      { !isLoading && (
        <>
          {/* Chat Header */}
          <div className="border-b border-gray-200 p-4">
            <h2 className="text-lg font-semibold text-gray-900">
              {chatTitle}
            </h2>
          </div>
        </>
      )}

      {isLoading && (
         <div className="flex-1 flex items-center justify-center">
           <div className="flex items-center space-x-2">
             <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
             <span className="text-gray-600">Loading messages...</span>
           </div>
         </div>
       )}

      { !isLoading && (
        <>
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
                {/* Render markdown content */}
                <div className="text-sm whitespace-pre-wrap">
                  <ReactMarkdown>{message.content}</ReactMarkdown>
                </div>
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
        </>
      )}

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

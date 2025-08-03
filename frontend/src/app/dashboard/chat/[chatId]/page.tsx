"use client"

import ChatInterface from "@/components/chat/chat-interface"
import { useAuth } from "@/lib/auth-context"
import { supabase } from "@/lib/supabase"
import { User } from "@supabase/supabase-js"
import { useEffect, useState } from "react"

type Message = {
  id: string
  content: string
  role: "user" | "ai"
  created_at: Date
}

export default function ChatPage({ params }: { params: { chatId: string } }) {
  const { user } = useAuth()
  const [accessToken, setAccessToken] = useState("")
  const [chatTitle, setChatTitle] = useState("")
  const [chatHistory, setChatHistory] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {

    const getAccessToken = async () => {
      const { data, error } = await supabase.auth.getSession()
      setAccessToken(data.session?.access_token || "")
      console.log('accessToken--------->>>>>>>>>>>>>', data.session?.access_token);

      const { data: chatSessionData, error: chatSessionError } = await supabase.from('chat_sessions').select('chat_title')
        .eq('id', params.chatId).single()
      if(chatSessionData) {
        setChatTitle(chatSessionData.chat_title || "")
      }
      console.log('chatSessionData--------->>>>>>>>>>>>>', chatSessionData);
      
      const {data: chatHistoryData, error: chatHistoryError} = await supabase.from('chat_history').select('*').eq('user_id', user?.id)
      .eq('session_id', params.chatId).order('created_at', { ascending: true })
      if(chatHistoryData) {
        setChatHistory(chatHistoryData as Message[])
      }
      console.log('chatHistoryData--------->>>>>>>>>>>>>', chatHistoryData);
      setIsLoading(false)
    }

    getAccessToken()
  }, [])

  return (
    <div className="h-full">
      <ChatInterface 
        chatId={params.chatId} 
        accessToken={accessToken} 
        user={user as User} 
        chatTitle={chatTitle} 
        chatHistory={chatHistory} 
        isLoading={isLoading}
      />
    </div>
  )
} 
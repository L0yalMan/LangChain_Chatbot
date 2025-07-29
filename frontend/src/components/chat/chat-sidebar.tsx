"use client"

import { useState, useEffect } from "react"
import { useRouter, useParams } from "next/navigation"
import { Plus, MessageCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { supabase } from "@/lib/supabase"

type Chat = {
  id: string
  chat_title: string
}

type User = {
  id: string
}

export default function ChatSidebar() {
  const router = useRouter()
  const params = useParams()
  const currentChatId = params?.chatId as string || ""
  const [user, setUser] = useState<User | null>(null)
  const [chats, setChats] = useState<Chat[]>([])
  useEffect(() => {
    const getUser = async () => {
      const { data: { user } } = await supabase.auth.getUser()
      if(user){
        const { data: chats, error } = await supabase.from('chat_sessions').select('*').eq('user_id', user.id)
        .order('created_at', { ascending: true })
        if(error) console.error('Error fetching chats:', error)
        else setChats(chats as Chat[])
        setUser(user as User)
        console.log(chats)
      }
    }
    
    getUser()
  }, [])

  const handleNewChat = async () => {
    router.push("/dashboard/chat-new")
    return
  }

  const handleChatClick = (chatId: string) => {
    router.push(`/dashboard/chat/${chatId}`)
  }

  return (
    <div className="w-80 bg-gray-50 border-r border-gray-200 flex flex-col h-full">
      <div className="p-4 border-b border-gray-200">
        <Button onClick={handleNewChat} className="w-full" disabled={!user}>
          <Plus className="mr-2 h-4 w-4" />
          New Chat
        </Button>
      </div>


      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">
          {chats.map((chat) => (
            <button
              key={chat.id}
              onClick={() => handleChatClick(chat.id)}
              className={`w-full text-left p-3 rounded-lg transition-colors ${
                currentChatId == chat.id
                  ? "bg-white shadow-sm border border-gray-200"
                  : "hover:bg-white hover:shadow-sm"
              }`}
            >
              <div className="flex items-start space-x-3">
                <MessageCircle className="h-5 w-5 text-gray-400 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">{chat.chat_title}</p>
                  {/* <p className="text-xs text-gray-500">{chat.lastMessage}</p> */}
                </div>
              </div>
            </button>
          ))}
        </div>
      </ScrollArea>
    </div>
  )
}

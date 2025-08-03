import ChatInterface from "@/components/chat/chat-interface"

export default function ChatPage() {
  return (
    <div className="h-full">
      <ChatInterface chatId="" accessToken="" user={null} chatTitle="" chatHistory={[]} isLoading={false} />
    </div>
  )
}

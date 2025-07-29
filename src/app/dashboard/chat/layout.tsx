import ChatSidebar from "@/components/chat/chat-sidebar"

export default function ChatLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="flex h-full">
      <ChatSidebar />
      <div className="flex-1 bg-white">
        {children}
      </div>
    </div>
  )
} 
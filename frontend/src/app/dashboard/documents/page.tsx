"use client"

import { useEffect, useState } from "react"
import { useAuth } from "@/lib/auth-context"
import { supabase } from "@/lib/supabase"
import { User } from "@supabase/supabase-js"
import DocumentManager from "@/components/documents/file-upload"

export default function DocumentsPage() {
  const { user } = useAuth()
  const [accessToken, setAccessToken] = useState("")

  useEffect(() => {
    const getAccessToken = async () => {
      const {data} = await supabase.auth.getSession()
      setAccessToken(data.session?.access_token || "")
      console.log('accessToken--------->>>>>>>>>>>>>', data.session?.access_token);
      console.log('user--------->>>>>>>>>>>>>', user);
    }

    getAccessToken()
  }, [])

  return (
    <div className="p-6 h-full overflow-auto">
      <div className="max-w-6xl mx-auto">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-gray-900">Documents</h1>
          <p className="text-gray-600">Upload documents and ingest websites for your chatbot</p>
        </div>
        <DocumentManager accessToken={accessToken} user={user as User} />
      </div>
    </div>
  )
}

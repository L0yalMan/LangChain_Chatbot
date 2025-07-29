"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Globe, Trash2, ExternalLink, CheckCircle, XCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useAuth } from "@/lib/auth-context"
import axios from "axios"
import { toast } from "@/hooks/use-toast"
import { supabase } from "@/lib/supabase"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"

type IngestedWebsite = {
  id: string
  url: string
  title: string
  domain: string
  ingestedAt: Date
  favicon?: string
}

export default function WebsiteIngestion() {
  const { user } = useAuth()
  const [websiteUrl, setWebsiteUrl] = useState("")
  const [isIngesting, setIsIngesting] = useState(false)
  const [ingestedWebsites, setIngestedWebsites] = useState<IngestedWebsite[]>([])
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [websiteToDelete, setWebsiteToDelete] = useState<IngestedWebsite | null>(null)

  useEffect(() => {
    const fetchIngestedWebsites = async () => {
      const {data: WebsiteLinkData, error} = await supabase.from("website_links").select("*").eq("user_id", user?.id).order("created_at", { ascending: false })

      if(error) {
        throw new Error(error.message)
      }

      setIngestedWebsites(WebsiteLinkData.map((website) => ({
        id: website.id,
        url: website.url,
        title: generateTitle(website.url),
        domain: extractDomain(website.url),
        ingestedAt: website.created_at,
        favicon: `https://www.google.com/s2/favicons?domain=${extractDomain(website.url)}&sz=32`,
      })))
    }
    fetchIngestedWebsites();
  }, [user?.id])

  const extractDomain = (url: string): string => {
    try {
      const urlObj = new URL(url.startsWith("http") ? url : `https://${url}`)
      return urlObj.hostname.replace("www.", "")
    } catch {
      return url
    }
  }

  const generateTitle = (url: string): string => {
    const domain = extractDomain(url)
    return domain.charAt(0).toUpperCase() + domain.slice(1).replace(/\.[^/.]+$/, "")
  }

  const handleIngestWebsite = async () => {
    if (!websiteUrl.trim()) return

    try {
      setIsIngesting(true)

      const response = await axios.post("https://f8c14eefce75.ngrok-free.app/ingest-website", {
        url: websiteUrl,
        userId: user?.id,
      })

      if(response.status === 200) {
        const {data: WebsiteLinkData, error} = await supabase.from("website_links").insert({ url: websiteUrl, user_id: user?.id }).select().single();

        if(error) {
          throw new Error(error.message)
        }

        const newWebsite: IngestedWebsite = {
          id: WebsiteLinkData.id,
          url: WebsiteLinkData.url,
          title: generateTitle(WebsiteLinkData.url),
          domain: extractDomain(WebsiteLinkData.url),
          ingestedAt: new Date(),
          favicon: `https://www.google.com/s2/favicons?domain=${extractDomain(WebsiteLinkData.url)}&sz=32`,
        }

        toast({
          title: "Success",
          description: response.data.message,
          variant: "default",
          icon: <CheckCircle className="h-5 w-5 text-green-600" />,
        })

        setIngestedWebsites((prev) => [newWebsite, ...prev])
        setWebsiteUrl("")
      } 
    }catch (error: any) {
      console.error("Error ingesting website:", error)
      toast({
        title: "Error",
        description: "An error occurred while ingesting the website",
        variant: "destructive",
        icon: <XCircle className="h-5 w-5 text-red-600" />,
      })
    } finally {
      setIsIngesting(false)
    }
  }

  const handleDeleteWebsite = (website: IngestedWebsite) => {
    setWebsiteToDelete(website)
    setDeleteDialogOpen(true)
  }

  const confirmDeleteWebsite = async () => {
    if (!websiteToDelete) return

    try {
      // Delete from Supabase
      const { error } = await supabase
        .from("website_links")
        .delete()
        .eq("id", websiteToDelete.id)
        .eq("user_id", user?.id)

      if (error) {
        throw new Error(error.message)
      }

      // Remove from local state
      setIngestedWebsites((prev) => prev.filter((website) => website.id !== websiteToDelete.id))
      
      toast({
        title: "Success",
        description: `${websiteToDelete.title} has been deleted`,
        variant: "default",
        icon: <CheckCircle className="h-5 w-5 text-green-600" />,
      })
    } catch (error: any) {
      console.error("Error deleting website:", error)
      toast({
        title: "Error",
        description: "Failed to delete website",
        variant: "destructive",
        icon: <XCircle className="h-5 w-5 text-red-600" />,
      })
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !isIngesting) {
      handleIngestWebsite()
    }
  }

  const isValidUrl = (url: string): boolean => {
    try {
      new URL(url.startsWith("http") ? url : `https://${url}`)
      return true
    } catch {
      return false
    }
  }

  return (
    <div className="space-y-6">
      {/* Website Ingestion Form */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Globe className="h-5 w-5" />
            <span>Ingest Website</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex space-x-3">
            <div className="flex-1">
              <Input
                type="url"
                placeholder="Enter website URL (e.g., https://example.com)"
                value={websiteUrl}
                onChange={(e) => setWebsiteUrl(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={isIngesting}
                className="w-full"
              />
            </div>
            <Button
              onClick={handleIngestWebsite}
              disabled={isIngesting || !websiteUrl.trim() || !isValidUrl(websiteUrl)}
              className="whitespace-nowrap"
            >
              {isIngesting ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Ingesting...
                </>
              ) : (
                "Ingest Website"
              )}
            </Button>
          </div>
          {websiteUrl && !isValidUrl(websiteUrl) && (
            <p className="text-sm text-red-500 mt-2">Please enter a valid URL</p>
          )}
          {/* Loading Bar */}
            {isIngesting && (
              <div className="mt-4">
                <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
                  <span>Ingesting website...</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden relative">
                  <div 
                    className="absolute top-0 left-0 h-full bg-blue-500 rounded-full"
                    style={{ 
                      width: "30%",
                      animation: "loading-stick 1.5s ease-in-out infinite"
                    }}
                  ></div>
                  <style jsx>{`
                    @keyframes loading-stick {
                      0% { left: -30%; }
                      100% { left: 100%; }
                    }
                  `}</style>
                </div>
              </div>
            )}
        </CardContent>
      </Card>

      {/* Ingested Websites List */}
      {ingestedWebsites.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Ingested Websites ({ingestedWebsites.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {ingestedWebsites.map((website) => (
                <div
                  key={website.id}
                  className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center space-x-4 flex-1 min-w-0">
                    {/* Favicon */}
                    <div className="flex-shrink-0">
                      {website.favicon ? (
                        <img
                          src={website.favicon || "/placeholder.svg"}
                          alt={`${website.domain} favicon`}
                          className="h-8 w-8 rounded"
                          onError={(e) => {
                            const target = e.target as HTMLImageElement
                            target.style.display = "none"
                            target.nextElementSibling?.classList.remove("hidden")
                          }}
                        />
                      ) : null}
                      <div
                        className={`h-8 w-8 bg-blue-100 rounded flex items-center justify-center ${website.favicon ? "hidden" : ""}`}
                      >
                        <Globe className="h-4 w-4 text-blue-600" />
                      </div>
                    </div>

                    {/* Website Info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2">
                        <h3 className="text-sm font-medium text-gray-900 truncate">{website.title}</h3>
                        <a
                          href={website.url.startsWith("http") ? website.url : `https://${website.url}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-gray-400 hover:text-gray-600"
                        >
                          <ExternalLink className="h-3 w-3" />
                        </a>
                      </div>
                      <p className="text-xs text-gray-500 truncate">{website.domain}</p>
                      <p className="text-xs text-gray-400">
                        Ingested {new Date(website.ingestedAt).toLocaleDateString()} at{" "}
                        {new Date(website.ingestedAt).toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </p>
                    </div>
                  </div>

                  {/* Delete Button */}
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => handleDeleteWebsite(website)}
                    className="text-gray-400 hover:text-red-500 flex-shrink-0"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Empty State */}
      {ingestedWebsites.length === 0 && (
        <Card>
          <CardContent className="p-8 text-center">
            <Globe className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No websites ingested yet</h3>
            <p className="text-gray-500">Enter a website URL above to start ingesting web content for your chatbot.</p>
          </CardContent>
        </Card>
      )}

      {/* Confirmation Dialog */}
      <ConfirmDialog
        open={deleteDialogOpen}
        onOpenChange={setDeleteDialogOpen}
        title="Delete Website"
        description={`Are you sure you want to delete "${websiteToDelete?.title}"? This action cannot be undone.`}
        confirmText="Delete"
        cancelText="Cancel"
        onConfirm={confirmDeleteWebsite}
        variant="destructive"
      />
    </div>
  )
}

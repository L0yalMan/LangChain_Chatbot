"use client"

import { useState, useCallback, useEffect } from "react"
import { useDropzone } from "react-dropzone"
import { Upload, X, FileText, File, CheckCircle, XCircle, FileTextIcon, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import axios from "axios"
import { useAuth } from "@/lib/auth-context"
import { toast } from "@/hooks/use-toast"
import WebsiteIngestion from "./website-ingestion"
import { supabase } from "@/lib/supabase"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"

type UploadedFile = {
  id: string
  type: string
  name: string
  size: number
  preview?: string
}

type SelectedFile = {
  id: string
  file: File
  preview?: string
}

const acceptedFileTypes = {
  "application/pdf": [".pdf"],
  "text/csv": [".csv"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
}

export default function FileUpload() {
  const { user } = useAuth()
  const [selectedFiles, setSelectedFiles] = useState<SelectedFile[]>([])
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [uploading, setUploading] = useState(false)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [fileToDelete, setFileToDelete] = useState<UploadedFile | null>(null)
  // const [uploadProgress, setUploadProgress] = useState(0)

  useEffect(() => {
    const fetchUploadedFiles = async () => {
      const {data: FileData, error} = await supabase.from("files").select("*").eq("user_id", user?.id).order("created_at", { ascending: false })

      if(error) {
        throw new Error(error.message)
      }

      setUploadedFiles(FileData.map((file) => ({
        id: file.id,
        type: file.file_type,
        name: file.file_name,
        size: file.file_size,
        preview: file.file_type.startsWith("image/") ? URL.createObjectURL(file) : undefined,
      })))
    }

    fetchUploadedFiles()
  }, [user?.id])

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles = acceptedFiles.map((file) => ({
      id: Math.random().toString(16).substring(2, 9),
      file,
      preview: file.type.startsWith("image/") ? URL.createObjectURL(file) : undefined,
    }))

    setSelectedFiles((prev) => [
      ...prev,
      ...newFiles.map((fileObj) => ({
        id: fileObj.id,
        file: fileObj.file,
        preview: fileObj.preview,
      })),
    ])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptedFileTypes,
    multiple: true,
  })

  const removeSelectedFile = (id: string) => {
    setSelectedFiles((prev) => prev.filter((file) => file.id !== id))
  }

  const removeUploadedFile = (id: string) => {
    setFileToDelete(uploadedFiles.find((file) => file.id === id) || null)
    setDeleteDialogOpen(true)
    setUploadedFiles((prev) => prev.filter((file) => file.id !== id))
  }


  const confirmDeleteWebsite = async () => {
    if (!fileToDelete) return

    try {
      // Delete from S3
      const response = await axios.post(`${process.env.NEXT_PUBLIC_BACKEND_HOST}/files/delete/`, { filename: fileToDelete.name, userId: user?.id })
      if(response.status !== 200) {
        throw new Error("Failed to delete file from S3")
      } 

      // Delete from Supabase
      const { error } = await supabase
        .from("files")
        .delete()
        .eq("id", fileToDelete.id)
        .eq("user_id", user?.id)

      if (error) {
        throw new Error(error.message)
      }

      // Remove from local state
      setUploadedFiles((prev) => prev.filter((file) => file.id !== fileToDelete.id))
      
      toast({
        title: "Success",
        description: `${fileToDelete.name} has been deleted`,
        variant: "default",
        icon: <CheckCircle className="h-5 w-5 text-green-600" />,
      })
    } catch (error: any) {
      console.error("Error deleting file:", error)
      toast({
        title: "Error",
        description: "Failed to delete file",
        variant: "destructive",
        icon: <XCircle className="h-5 w-5 text-red-600" />,
      })
    }
  }

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return

    setUploading(true)
    // setUploadProgress(0)

    try {
      await Promise.all(selectedFiles.map(async (selectedFile) => {
        const formData = new FormData()
        formData.append("file", selectedFile.file as unknown as Blob)
        formData.append("user_id", user?.id as string)
        
        try {
          const response = await axios.post(`${process.env.NEXT_PUBLIC_BACKEND_HOST}/files/upload`, formData)
          if(response.status === 200) {
            
            const {data: FileData, error} = await supabase.from("files").insert({
              file_name: selectedFile.file.name,
              file_type: selectedFile.file.type,
              file_size: selectedFile.file.size,
              user_id: user?.id,
            }).select().single();

            if(error) {
              throw new Error(error.message)
            }

            setUploadedFiles((prev) => [...prev, {
              id: FileData.id,
              type: FileData.file_type,
              name: FileData.file_name,
              size: FileData.file_size,
              preview: selectedFile.preview,
            }])

            setSelectedFiles((prev) => prev.filter((f) => f.id !== selectedFile.id))
            toast({
              title: "Success",
              description: `${FileData.file_name} uploaded successfully`,
              variant: "default",
              icon: <CheckCircle className="h-5 w-5 text-green-600" />,
            })
          } else {
            throw new Error("Upload failed")
          }
        } catch (error) {
          console.error("Failed to upload file:", selectedFile.file.name, error)
          toast({
            title: "Error",
            description: `Failed to upload ${selectedFile.file.name}`,
            variant: "destructive",
            icon: <XCircle className="h-5 w-5 text-red-600" />,
          })
        }
      }))
    } catch (error) {
      console.error("Upload process failed:", error)
      toast({
        title: "Error",
        description: "Upload process failed",
        variant: "destructive",
        icon: <XCircle className="h-5 w-5 text-red-600" />,
      })
    }

    setUploading(false)
    // setUploadProgress(0)
  }

  const getFileIcon = (fileType: string) => {
    if (fileType.includes("pdf")) return <FileText className="h-8 w-8 text-red-500" />
    if (fileType.includes("csv")) return <File className="h-8 w-8 text-green-500" />
    if (fileType.includes("docx")) return <FileText className="h-8 w-8 text-blue-500" />
    return <File className="h-8 w-8 text-gray-500" />
  }

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <Card>
        <CardContent className="p-6">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive ? "border-blue-400 bg-blue-50" : "border-gray-300 hover:border-gray-400"
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            {isDragActive ? (
              <p className="text-blue-600">Drop the files here...</p>
            ) : (
              <div>
                <p className="text-gray-600 mb-2">Drag & drop files here, or click to select files</p>
                <p className="text-sm text-gray-500">Supports PDF, CSV files</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Selected Files Preview */}
      {selectedFiles.length > 0 && (
        <Card>
          <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <FileTextIcon className="h-5 w-5" />
            <span>Selected Files ({selectedFiles.length})</span>
          </CardTitle>
        </CardHeader>
          <CardContent className="p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="text-ml text-gray-500">Supported file types: PDF, CSV</div>
              <Button onClick={handleUpload} disabled={uploading}>
                {uploading ? "Uploading..." : "Upload Files"}
              </Button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {selectedFiles.map((uploadedFile) => (
                <div key={uploadedFile.id} className="border rounded-lg p-4 relative">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => removeSelectedFile(uploadedFile.id)}
                    className="absolute top-2 right-2 h-6 w-6"
                    disabled={uploading}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                  <div className="flex items-center space-x-3">
                    {getFileIcon(uploadedFile.file.type)}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{uploadedFile.file.name}</p>
                      <p className="text-xs text-gray-500">{(uploadedFile.file.size / 1024 / 1024).toFixed(2)} MB</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            
            {/* Loading Bar */}
            {uploading && (
              <div className="mt-4">
                <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
                  <span>Uploading files...</span>
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
      )}

      {/* Uploaded Files */}
      {uploadedFiles.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Uploaded Documents ({uploadedFiles.length})</CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {uploadedFiles.map((uploadedFile) => (
                <div key={uploadedFile.id} className="border rounded-lg p-4 relative flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getFileIcon(uploadedFile.type)}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{uploadedFile.name}</p>
                      <p className="text-xs text-gray-500">{(uploadedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                      <p className="text-xs text-green-600">Uploaded</p>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => removeUploadedFile(uploadedFile.id)}
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
      {uploadedFiles.length === 0 && (
        <Card>
          <CardContent className="p-8 text-center">
            <FileTextIcon className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No documents uploaded yet</h3>
            <p className="text-gray-500">Upload documents to start chatting with your chatbot.</p>
          </CardContent>
        </Card>
      )}

      {/* Website Ingestion Section */}
      <WebsiteIngestion />

      {/* Confirmation Dialog */}
      <ConfirmDialog
        open={deleteDialogOpen}
        onOpenChange={setDeleteDialogOpen}
        title="Delete File"
        description={`Are you sure you want to delete "${fileToDelete?.name}"? This action cannot be undone.`}
        confirmText="Delete"
        cancelText="Cancel"
        onConfirm={confirmDeleteWebsite}
        variant="destructive"
      />
    </div>
  )
}

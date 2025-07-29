import DocumentManager from "@/components/documents/file-upload"

export default function DocumentsPage() {
  return (
    <div className="p-6 h-full overflow-auto">
      <div className="max-w-6xl mx-auto">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-gray-900">Documents</h1>
          <p className="text-gray-600">Upload documents and ingest websites for your chatbot</p>
        </div>
        <DocumentManager />
      </div>
    </div>
  )
}

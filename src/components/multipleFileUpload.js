import React from 'react';

const MultipleFileUpload = ({ selectedFiles, setSelectedFiles, handleUploadFile }) => {
  const handleFileChange = (e) => {
    if (e.target.files) {
      setSelectedFiles([...selectedFiles, ...Array.from(e.target.files)]);
    }
  };

  return (
    <div className="max-w-md mx-auto">
      <div className="mt-8">
        <label className="block">
          <span className="text-gray-700">Choose Files</span>
          <input
            type="file"
            onChange={handleFileChange}
            className="block w-full text-sm text-slate-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-violet-50 file:text-violet-700
              hover:file:bg-violet-100"
            multiple
          />
        </label>
      </div>
      <button
        onClick={handleUploadFile}
        className="mt-4 bg-blue-500 hover:bg-blue-700 text-white py-2 px-4 rounded"
      >
        Upload
      </button>
      {selectedFiles.length > 0 && (
        <div className="mt-4">
          <p className="font-semibold">Selected Files:</p>
          <ul>
            {selectedFiles.map((file, index) => (
              <li key={index}>{file.name}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default MultipleFileUpload;
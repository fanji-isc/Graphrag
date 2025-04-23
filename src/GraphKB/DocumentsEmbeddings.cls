/// adopted from VectorDBDemo
/// Hackathon 2025 Team 18
Class GraphKB.DocumentsEmbeddings Extends %Persistent
{

Property Embedding As %Library.Vector(DATATYPE = "FLOAT", LEN = 384);

// LEN will depend on the size of the vector output of the embedding model

ClassMethod LoadData(filepath As %Library.String, purge As %Library.Integer = 1) As %Status
{
    // purge - Whether to clean the table of previously loaded data; default to 1.
    if purge { // purge data
        set myquery = "TRUNCATE TABLE GraphKB.DocumentsEmbeddings"
        set tStatement = ##class(%SQL.Statement).%New()
        $$$ThrowOnError(tStatement.%Prepare(myquery))
        set rset = tStatement.%Execute()
    }

    set myquery = "LOAD DATA FROM FILE '"_filepath_"' INTO GraphKB.DocumentsEmbeddings USING {""from"":{""file"":{""header"":true}}}"
    set tStatement = ##class(%SQL.Statement).%New()
    $$$ThrowOnError(tStatement.%Prepare(myquery))
    set rset = tStatement.%Execute()
    if (rset.%SQLCODE < 0) {
        throw ##class(%Exception.SQL).CreateFromSQLCODE(rset.%SQLCODE, rset.%Message)
    }
    return rset
}

ClassMethod Insert(embedding As %String) As %Status
{
    // Inserts row into embedding table using class object
    set sc = $$$OK
    try {
        set newObj = ..%New()
        // We use the DisplayToLogical function of the property to take in a string representation of the vector. The string should include brackets eg. embedding = "[1,2,3]"
        set newObj.Embedding = ..EmbeddingDisplayToLogical(embedding) 
        do newObj.%Save()
    }   
    catch e {
        set sc = e.AsStatus()
    }
    quit sc
}

ClassMethod DataToEmbeddings(batchSize = 128, sourceField = "abstract", sourceTable = "Documents") As %Status
{
    // Converts Data to Embeddings 
    set sc = $$$OK
    try {
        // Init list for later
        set builtins = ##class(%SYS.Python).Builtins() 
        set entityList = builtins.list() 

        // Get all Descriptions from downloads table
        set myquery = "SELECT "_sourceField_" from GraphKB."_sourceTable
        set tStatement = ##class(%SQL.Statement).%New()
        $$$ThrowOnError(tStatement.%Prepare(myquery))
        set rset = tStatement.%Execute()
        if (rset.%SQLCODE < 0) {
            throw ##class(%Exception.SQL).CreateFromSQLCODE(rset.%SQLCODE, rset.%Message)
        }

        // Initialize counters
        set idx = 0
        set batchIdx = 0

        while rset.%Next(){
            // Print out progress
            if (idx # 1000 = 0){
                write !, "Result Set idx: "_ idx
            }
            set idx = idx + 1

            // Add the descriptions to batches
            set batchIdx = batchIdx + 1
            do entityList.append(rset.%Get("entityid"))

            // Convert to embedding in batches, as it's more efficient
            if (batchIdx = batchSize){
                set embeddingList = ..GetEmbeddingsBatchPy(entityList)
                for j = 0:1:(batchIdx-1){
                    set st = ..Insert(embeddingList."__getitem__"(j)) 
                    $$$ThrowOnError(st)
                }
                set entitynList = builtins.list() 
                set batchIdx = 0
            }
        }
        // Handle leftovers from batch
        if (batchIdx '= 0) {
            for j = 0:1:(batchIdx-1){
                    set st = ..Insert(embeddingList."__getitem__"(j)) 
                    $$$ThrowOnError(st)
                }
        }
    }
    catch e {
        set sc = e.AsStatus()
        zw sc
    }
    quit sc
}

ClassMethod GetEmbeddingsBatchPy(sentences) [ Language = python ]
{
    # Converts a list of text into a list of embeddings
    import json
    import numpy as np
    import sentence_transformers
    
    # Get embeddings using sentence_transformers
    # If using windows, set your own cache folder: model = sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='C:\\InterSystems\\IRIS_LLM\\Mgr\\huggingface_cache')
    model = sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)

    # Round
    embeddings_np = np.round(embeddings, 7)

    # Convert numpy arrays to list of strings
    embeddings_str_list = [str(embedding.tolist()) for embedding in embeddings_np]
    return embeddings_str_list
}

ClassMethod GetEmbeddingsPy(text) [ Language = python ]
{
    # Converts a single text to embeddings. 
    import numpy as np
    import sentence_transformers

    # Get embedding.  Set your own cache folder,
    # If using windows, set your own cache folder: model = sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='C:\\InterSystems\\IRIS_LLM\\Mgr\\huggingface_cache')
    model = sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode([text])[0]
    
    # Round
    embeddings_np = np.round(embeddings, 7)

    # Convert numpy array to string
    return str(embeddings_np.tolist())
}

Storage Default
{
<Data name="DocumentsEmbeddingsDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>Embedding</Value>
</Value>
</Data>
<DataLocation>^GraphKB.DocumentsEmbeddingsD</DataLocation>
<DefaultData>DocumentsEmbeddingsDefaultData</DefaultData>
<ExtentSize>159</ExtentSize>
<IdLocation>^GraphKB.DocumentsEmbeddingsD</IdLocation>
<IndexLocation>^GraphKB.DocumentsEmbeddingsI</IndexLocation>
<Property name="%%CLASSNAME">
<AverageFieldSize>2</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,0,16,$lb("-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000"),$lb(21,21,21,21,21,21,21,21,21,21,21,21,21,21,21),$lb(758198320,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,758198320))</Histogram>
<OutlierSelectivity>.999999:</OutlierSelectivity>
<Selectivity>0.0001%</Selectivity>
</Property>
<Property name="%%ID">
<AverageFieldSize>3</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,1,0,$lb(1,11,22,33,44,55,66,77,88,99,109,119,129,139,149,159),$lb(1,0,0,0,0,0,0,0,0,0,1,1,1,1,1),"")</Histogram>
<Selectivity>1</Selectivity>
</Property>
<Property name="Embedding">
<AverageFieldSize>4420</AverageFieldSize>
<Selectivity>1</Selectivity>
</Property>
<SQLMap name="IDKEY">
<BlockCount>-640</BlockCount>
</SQLMap>
<StreamLocation>^GraphKB.DocumentsEmbeddingsS</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}

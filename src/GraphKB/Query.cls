Class GraphKB.Query Extends %RegisteredObject
{

ClassMethod Search(userQuery As %String, graphSearchCount As %Integer = 0, semanticSearchCount As %Integer = 0) As %String
{
    set sc = $$$OK
    if (graphSearchCount <= 0) && (semanticSearchCount <=0) {
        write "Nothing to display. You need to supply at least 1 non-zero count for graph search or semantic search."
        return '$$$OK
    } else {
        write "Displaying at most "_graphSearchCount_" counts of graph search and "_semanticSearchCount_" counts of semantic search."_$CHAR(10)

    }
    try {
        // Set up SQL Query to get whatever fields you want, joined on the embeddings
        if graphSearchCount > 0 { // Graph Search Enabled
            set myEntityQuery = "SELECT TOP ? entity.entityid FROM GraphKB.EntityEmbeddings emb JOIN GraphKB.Entity entity ON emb.ID = entity.ID ORDER BY VECTOR_DOT_PRODUCT(emb.Embedding, TO_VECTOR(?)) DESC"
            set tEntityStatement = ##class(%SQL.Statement).%New()
            $$$ThrowOnError(tEntityStatement.%Prepare(myEntityQuery))

            // Convert query string into embedding
            set userQueryEntity = $CLASSMETHOD("GraphKB.EntityEmbeddings","GetEmbeddingsPy",userQuery)
            
            // Execute SQL
            set rsetEntity = tEntityStatement.%Execute(graphSearchCount, userQueryEntity)
            if (rsetEntity.%SQLCODE < 0) {
                throw ##class(%Exception.SQL).CreateFromSQLCODE(rsetEntity.%SQLCODE, rsetEntity.%Message)
            }

            // If you're implementing RAG, you would format your retrieved information here before passing the context to an LLM
            while rsetEntity.%Next(){
                set entityList = $s($d(entityList):entityList,1:"")_$lb(rsetEntity.%Get("entityid"))    
            }
            set myquery = "SELECT TOP ? docs.abstract FROM GraphKB.Relations rel JOIN GraphKB.Documents docs ON rel.docid=docs.docid WHERE rel.source %INLIST ? OR rel.target %INLIST ? GROUP BY rel.docid ORDER by count(rel.docid) DESC UNION SELECT TOP ? docs.abstract FROM GraphKB.DocumentsEmbeddings emb JOIN GraphKB.Documents docs ON emb.ID = docs.ID ORDER BY VECTOR_DOT_PRODUCT(emb.Embedding, TO_VECTOR(?)) DESC"
        } elseif semanticSearchCount > 0 { // Semantic search 
            set myquery = "SELECT TOP ? docs.abstract FROM GraphKB.DocumentsEmbeddings emb JOIN GraphKB.Documents docs ON emb.ID = docs.ID ORDER BY VECTOR_DOT_PRODUCT(emb.Embedding, TO_VECTOR(?)) DESC"

        }
        set tStatement = ##class(%SQL.Statement).%New()
        $$$ThrowOnError(tStatement.%Prepare(myquery))

        // Convert query string into embedding
        set userQueryEmbedding = $CLASSMETHOD("GraphKB.DocumentsEmbeddings","GetEmbeddingsPy",userQuery)
        
        // Execute SQL
        //graphSearchCount,userQueryEmbedding
        set rset = $s($d(entityList):tStatement.%Execute(graphSearchCount,entityList,entityList,semanticSearchCount,userQueryEmbedding),1:tStatement.%Execute(semanticSearchCount, userQueryEmbedding))
        if (rset.%SQLCODE < 0) {
            throw ##class(%Exception.SQL).CreateFromSQLCODE(rset.%SQLCODE, rset.%Message)
        }
        set retrievedInfo = "", resultCount=0
        // If you're implementing RAG, you would format your retrieved information here before passing the context to an LLM
        while rset.%Next(){
            set resultCount = resultCount + 1
            set retrievedInfo = retrievedInfo _ $C(13,10) _ " Abstract "_resultCount_": " _rset.%Get("abstract")_$CHAR(10)
        }
        write !, retrievedInfo      
        return retrievedInfo

    }
    catch e {
        set sc = e.AsStatus()
        zw sc
        return sc 
    }
}

}

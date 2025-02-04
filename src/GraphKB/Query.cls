Class GraphKB.Query Extends %RegisteredObject
{

ClassMethod GetTopK(kCount As %Integer, userQuery As %String, source As %String = "Entity") As %String
{
    set sc = $$$OK
    try {
        set sourceField=$s(source="Entity":"entityid",source="Documents":"abstract")
        
        // Set up SQL Query to get whatever fields you want, joined on the embeddings
        if source="Entity" { // Graph Search
            set myEntityQuery = "SELECT TOP ? entity.entityid FROM GraphKB.EntityEmbeddings emb JOIN GraphKB.Entity entity ON emb.ID = entity.ID ORDER BY VECTOR_DOT_PRODUCT(emb.Embedding, TO_VECTOR(?)) DESC"
            set tEntityStatement = ##class(%SQL.Statement).%New()
            $$$ThrowOnError(tEntityStatement.%Prepare(myEntityQuery))

            // Convert query string into embedding
            set userQueryEntity = $CLASSMETHOD("GraphKB.EntityEmbeddings","GetEmbeddingsPy",userQuery)
            
            // Execute SQL
            set rsetEntity = tEntityStatement.%Execute(kCount, userQueryEntity)
            if (rsetEntity.%SQLCODE < 0) {
                throw ##class(%Exception.SQL).CreateFromSQLCODE(rsetEntity.%SQLCODE, rsetEntity.%Message)
            }

            // If you're implementing RAG, you would format your retrieved information here before passing the context to an LLM
            while rsetEntity.%Next(){
                set entityList = $s($d(entityList):entityList,1:"")_$lb(rsetEntity.%Get("entityid"))    
            }
            set myquery = "SELECT TOP ? docs.abstract FROM GraphKB.Relations rel JOIN GraphKB.Documents docs ON rel.docid=docs.docid WHERE rel.source %INLIST ? OR rel.target %INLIST ? GROUP BY rel.docid ORDER by count(rel.docid) DESC " /// UNION SELECT TOP ? docs.abstract FROM GraphKB.DocumentsEmbeddings emb JOIN GraphKB.Documents docs ON emb.ID = docs.ID ORDER BY VECTOR_DOT_PRODUCT(emb.Embedding, TO_VECTOR(?)) DESC"
        } elseif source = "Documents" { // Semantic search 
            set myquery = "SELECT TOP ? docs.abstract FROM GraphKB.DocumentsEmbeddings emb JOIN GraphKB.Documents docs ON emb.ID = docs.ID ORDER BY VECTOR_DOT_PRODUCT(emb.Embedding, TO_VECTOR(?)) DESC"

        }
        set tStatement = ##class(%SQL.Statement).%New()
        $$$ThrowOnError(tStatement.%Prepare(myquery))

        // Convert query string into embedding
        set userQueryEmbedding = $CLASSMETHOD("GraphKB.DocumentsEmbeddings","GetEmbeddingsPy",userQuery)
        
        // Execute SQL
        //kCount,userQueryEmbedding
        set rset = $s(source="Entity":tStatement.%Execute(kCount,entityList,entityList),source="Documents":tStatement.%Execute(kCount, userQueryEmbedding))
        if (rset.%SQLCODE < 0) {
            throw ##class(%Exception.SQL).CreateFromSQLCODE(rset.%SQLCODE, rset.%Message)
        }
        set retrievedInfo = "" 
        // If you're implementing RAG, you would format your retrieved information here before passing the context to an LLM
        while rset.%Next(){
            set retrievedInfo = retrievedInfo _ $C(13,10) _ " abstract: " _rset.%Get("abstract")_$CHAR(10)
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

Class GraphKB.Relations Extends %Persistent
{

Property docid As %Library.String(MAXLEN = 200);

Property source As %Library.String(MAXLEN = 1000);

Property sourcetype As %Library.String(MAXLEN = 1000);

Property target As %Library.String(MAXLEN = 1000);

Property targettype As %Library.String(MAXLEN = 1000);

Property type As %Library.String(MAXLEN = 1000);

ClassMethod LoadData(filepath As %Library.String, purge As %Library.Integer = 1) As %Status
{
    // purge - Whether to clean the table of previously loaded data; default to 1.
    if purge { // purge data
        set myquery = "TRUNCATE TABLE GraphKB.Relations"
        set tStatement = ##class(%SQL.Statement).%New()
        $$$ThrowOnError(tStatement.%Prepare(myquery))
        set rset = tStatement.%Execute()
    }

    set myquery = "LOAD DATA FROM FILE '"_filepath_"' INTO GraphKB.Relations USING {""from"":{""file"":{""header"":true}}}"
    set tStatement = ##class(%SQL.Statement).%New()
    $$$ThrowOnError(tStatement.%Prepare(myquery))
    set rset = tStatement.%Execute()
    if (rset.%SQLCODE < 0) {
        throw ##class(%Exception.SQL).CreateFromSQLCODE(rset.%SQLCODE, rset.%Message)
    }
    return rset
}

Storage Default
{
<Data name="RelationsDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>docid</Value>
</Value>
<Value name="3">
<Value>source</Value>
</Value>
<Value name="4">
<Value>sourcetype</Value>
</Value>
<Value name="5">
<Value>target</Value>
</Value>
<Value name="6">
<Value>targettype</Value>
</Value>
<Value name="7">
<Value>type</Value>
</Value>
</Data>
<DataLocation>^GraphKB.RelationsD</DataLocation>
<DefaultData>RelationsDefaultData</DefaultData>
<ExtentSize>142</ExtentSize>
<IdLocation>^GraphKB.RelationsD</IdLocation>
<IndexLocation>^GraphKB.RelationsI</IndexLocation>
<Property name="%%CLASSNAME">
<AverageFieldSize>2</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,0,16,$lb("-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000","-10000000000000000000"),$lb(21,21,21,21,21,21,21,21,21,21,21,21,21,21,21),$lb(758198320,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,758198320))</Histogram>
<OutlierSelectivity>.999999:</OutlierSelectivity>
<Selectivity>0.0001%</Selectivity>
</Property>
<Property name="%%ID">
<AverageFieldSize>3</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,1,0,$lb(1,10,20,30,40,50,60,70,79,88,97,106,115,124,133,142),$lb(1,0,0,0,0,0,0,1,0,0,0,1,1,1,1),"")</Histogram>
<Selectivity>1</Selectivity>
</Property>
<Property name="docid">
<AverageFieldSize>3.51</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,0,0,$lb(" 0"," 1"," 10"," 12"," 14"," 15"," 16"," 17"," 18"," 2"," 3"," 5"," 5"," 7"," 8"," 9"),$lb(1,2,2,2,2,2,2,2,1,1,1,3,1,1,1),$lb(540016640,805306368,822083584,0,805306368,805306368,838860800,838860800,872415232,872415232,889192448,889192448,905969664,905969664,922746880,922746880,939524096,825753600,838860800,838860800,855638016,855638016,889192448,0,0,889192448,922746880,922746880,939524096,939524096,956301312,540606464))</Histogram>
<Selectivity>5.0000%</Selectivity>
</Property>
<Property name="source">
<AverageFieldSize>29.07</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,0,0,$lb(" 50% NEUTRALIZING ANTIBODY TITER"," AN IMMUNOLOGICAL AUTOBIOGRAPHY"," CLINICAL TRIAL OUTCOME (CTO) DATASET"," CORRELATES OF RISK"," CYD15"," DEMO"," ETHAN B LUDMIR"," JACOB APTEKAR"," MARCO CARONE"," NLP"," PHASE I TRIALS"," SLEEP MONITORING ALGORITHM"," SLEEP MONITORING ALGORITHM"," TRIAL2VEC"," TXSHIFT R PACKAGE"," ZIFENG WANG"),$lb(1,1,2,2,1,1,1,1,1,1,1,28,1,2,1),$lb(540356645,892347680,1095639113,1095639113,1129072974,1279872585,1330795077,1330795077,1497641269,1129923633,1145392463,1145392463,1163151425,1163151425,1245791055,1245791055,1296126531,1296126531,1313624064,1313624064,1346912595,1346912595,1397507397,0,0,1397507397,1414678849,1380532556,1481853001,1415074632,1514751557,542787910))</Histogram>
<OutlierSelectivity>.070423:"Sleep Monitoring Algorithm"</OutlierSelectivity>
<Selectivity>1.1767%</Selectivity>
</Property>
<Property name="sourcetype">
<AverageFieldSize>7.3</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,0,0,$lb(" AUTHOR"," AUTHOR"," AUTHOR"," AUTHOR"," AUTHOR"," PAPER"," PAPER"," PAPER"," TOPIC"," TOPIC"," TOPIC"," TOPIC"," TOPIC"," TOPIC"," TOPIC"," TOPIC"),$lb(8,8,8,8,1,7,7,1,7,7,7,7,7,7,7),$lb(541152596,0,0,0,0,0,0,0,0,1096111176,1346457669,0,0,0,0,1346457669,1414484041,0,0,0,0,0,0,0,0,0,0,0,0,0,0,542396240))</Histogram>
<Selectivity>33.3333%</Selectivity>
</Property>
<Property name="target">
<AverageFieldSize>20.3</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,0,0,$lb(" 50% NEUTRALIZING ANTIBODY TITER"," AN IMMUNOLOGICAL AUTOBIOGRAPHY"," CHUFAN GAO"," CONTROLLED-RISK CURVE"," CYD14"," ELIZABETH H. SLATE"," HIV TREATMENT"," JIMENG SUN"," MATS STENSRUD"," PANACEA"," PETER F. THALL"," SHIVASHANKAR THATI"," TRIALSYNTH"," TXSHIFT R PACKAGE"," VIROLOGIC FAILURE"," ZIQIAN GENG"),$lb(1,1,2,2,1,1,1,1,1,2,1,1,2,1,1),$lb(540356645,892347680,1095639113,1095639113,1128813894,1213548097,1330533458,1330533458,1497641268,1129923633,1162627418,1162627418,1212765728,1212765728,1246317893,1246317893,1296127059,1296127059,1346457153,1095647555,1163150674,1346720837,1397246294,1397246294,1414678849,1380532556,1481853001,1415074632,1447645775,1447645775,1514754377,542787921))</Histogram>
<Selectivity>1.0309%</Selectivity>
</Property>
<Property name="targettype">
<AverageFieldSize>7.23</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,0,0,$lb(" AUTHOR"," AUTHOR"," AUTHOR"," AUTHOR"," PAPER"," TOPIC"," TOPIC"," TOPIC"," TOPIC"," TOPIC"," TOPIC"," TOPIC"," TOPIC"," TOPIC"," TOPIC"," TOPIC"),$lb(8,8,8,1,1,7,7,7,7,7,7,7,7,7,7),$lb(541152596,0,0,0,0,0,0,1096111176,1346457669,1346457669,1414484041,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,542396240))</Histogram>
<Selectivity>33.3333%</Selectivity>
</Property>
<Property name="type">
<AverageFieldSize>10.8</AverageFieldSize>
<Histogram>$lb(.06666666666666666667,0,0,$lb(" AUTHORED"," AUTHORED"," AUTHORED"," AUTHORED"," AUTHORED"," AUTHORED"," AUTHORED"," AUTHORED"," DISCUSSES"," DISCUSSES"," RELATED_TO"," RELATED_TO"," RELATED_TO"," RELATED_TO"," RELATED_TO"," RELATED_TO"),$lb(10,10,10,10,10,10,10,1,11,1,12,12,12,12,12),$lb(541152596,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1096111176,1145656131,0,0,1145656131,1380273217,0,0,0,0,0,0,0,0,0,0,542262604))</Histogram>
<Selectivity>33.3333%</Selectivity>
</Property>
<SQLMap name="IDKEY">
<BlockCount>-8</BlockCount>
</SQLMap>
<StreamLocation>^GraphKB.RelationsS</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}

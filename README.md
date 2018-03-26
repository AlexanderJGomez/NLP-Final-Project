# NLP-Final-Project

To run, call createSummary with a file name, optionally specifying the separator between sentences (uses new line as default).

Requires numpy, and nltk.

ROUGE algorithm from: https://github.com/RxNLP/ROUGE-2.0

To run ROUGE: java -jar rouge.jar

That runs using the human summaries in /reference and the system summaries in /system

It then puts the results in results.csv

Note: Rouge matches the summaries in both folders using task_name  in : "{task_name}_{does not matter what this is}.txt"

For example: reference/newsarticle1_summary.txt and system/newsarticle1_syssum.txt would be compared
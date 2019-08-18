# Merge Conflicts and Refactorings

### Dumped data collected by https://github.com/Symbolk/RefactoringsInMergeCommits on the following projects (Apri 4, 2019):

| Name              | URL               |
| ----------------- | ----------------- |
| 	realm-java	          | https://github.com/realm/realm-java   |
| junit          | https://github.com/junit-team/junit |
| javaparser| https://github.com/javaparser/javaparser   |
| gradle             |https://github.com/gradle/gradle |
| error-prone              |https://github.com/google/error-prone    |
| elasticsearch     | https://github.com/elasticsearch/elasticsearch	       |
|deeplearning4j    |https://github.com/Symbolk/deeplearning4j         |
|  storm        | https://github.com/apache/storm      |
| cassandra|   https://github.com/apache/cassandra |
| antlr4       | https://github.com/antlr/antlr4   |

> Since the original repo of deeplearning4j (https://github.com/eclipse/deeplearning4j) was dramatically changed in July, 2019, the commit history was totally rewritten, so please use the forked repo instead. 



## Requirements

1. MySQL 5.7

## Import Data

1. Start MySQL server;

2. Open terminal and connect to the database:

   ```
   mysql -u root -p
   ```

3. Create a database named `refactoring_analysis`:

   ```
   CREATE DATABASE refactoring_analysis;
   ```

4. Import the data:

   ```
   USE refactoring_analysis
   source 20190401.sql
   ```

   ## Inspect Data

Inspect the data with the following example SQL statements, in MySQL workbench or just terminal:

```
SHOW databases;
USE refactoring_analysis;
SHOW tables;
SELECT * FROM project;
SELECT * FROM project WHERE name="error-prone";
SELECT * FROM merge_commit WHERE project_id=10 AND is_conflicting=1 AND timestamp < 1554048000;

SELECT * FROM refactoring WHERE project_id=23;
SELECT * FROM refactoring_region WHERE project_id=23 AND refactoring_id=8782;

SELECT count(*) FROM conflicting_region WHERE project_id=23;
```


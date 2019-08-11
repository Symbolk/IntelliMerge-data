show databases;

use refactoring_analysis;

use test;

show tables;

describe merge_commit;

select * from project;

select * from project where name="error-prone";

select count(*) from merge_commit WHERE project_id=11 and is_conflicting=1;

select count(*) from merge_commit WHERE project_id=10 and is_conflicting=1 and timestamp < 1554048000;

select * from merge_commit WHERE project_id=11;

select * from merge_commit WHERE project_id=23;



select * from refactoring where project_id=23 and refactoring_commit_id=8782;

select * from refactoring_region where project_id=23;

select * from refactoring_region where project_id=23 and refactoring_commit_id=8782;

select count(*) from conflicting_region where project_id=;
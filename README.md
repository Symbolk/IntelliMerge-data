# wrong-merge-commit-examples

A collection of wrongly merged&amp;committed examples with obvious syntax errors, spotted from real projects. Each folder represents one project, inside it are subfolders named with full commitId. The `summary.txt` records each file's relative path and its problem.

Since reproduce the compile&build environment is hard and most of the studied projects do not use a CI system, we collected the examples here simply by checking for syntax errors. 

We believe there are more wrongly merged files at the merge commit, which proves it not to be a good oracle.


## Reproduce

All examples can be seen in the cloned repo of the corresponding project in the following steps, take the `realm-java` for example:

1. Clone the `realm-java` from https://github.com/realm/realm-java;

2. Under the cloned repo, run the following command in shell/bash:

```
git show COMMITID:PATH
```

> COMMITID: the commitId of merge commit that contains wrongly merged files. PATH: the relative path of the wrongly merged file.

For example:

```
git show 7eae50c03:realm/realm-annotations-processor/src/main/java/io/realm/processor/RealmProxyClassGenerator.java
```

3. Check the line number of the reported problem in summary.txt and jump to it.

For example: Jump to line 852 of the the above command's output, you will see `.nextControlFlow("else");` does not have a caller.

```java
        writer.beginControlFlow("if (object instanceof RealmObjectProxy && ((RealmObjectProxy)object).realmGet$proxyState().getRealm$realm() != null && ((RealmObjectProxy)object).realmGet$proxyState().getRealm$realm().getPath().equals(realm.getPath()))");
                writer.emitStatement("cache.put(object, ((RealmObjectProxy)object).realmGet$proxyState().getRow$realm().getIndex())");
                .nextControlFlow("else");
                addPrimaryKeyCheckIfNeeeded(metadata, true, writer);
                writer.endControlFlow();
```




package io.realm.processor;

import com.squareup.javawriter.JavaWriter;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;

import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.DeclaredType;
import javax.lang.model.util.Types;
import javax.tools.JavaFileObject;

public class RealmProxyClassGenerator {
    private ProcessingEnvironment processingEnvironment;
    private ClassMetaData metadata;
    private final String simpleClassName;
    private final String qualifiedClassName;
    private final String interfaceName;
    private final String qualifiedGeneratedClassName;

    public RealmProxyClassGenerator(ProcessingEnvironment processingEnvironment, ClassMetaData metadata) {
        this.processingEnvironment = processingEnvironment;
        this.metadata = metadata;
        this.simpleClassName = metadata.getSimpleClassName();
        this.qualifiedClassName = metadata.getFullyQualifiedClassName();
        this.interfaceName = Utils.getProxyInterfaceName(simpleClassName);
        this.qualifiedGeneratedClassName = String.format("%s.%s",
                Constants.REALM_PACKAGE_NAME, Utils.getProxyClassName(simpleClassName));
    }

    public void generate() throws IOException, UnsupportedOperationException {
        JavaFileObject sourceFile = processingEnvironment.getFiler().createSourceFile(qualifiedGeneratedClassName);
        JavaWriter writer = new JavaWriter(new BufferedWriter(sourceFile.openWriter()));


        writer.setIndent(Constants.INDENT);

        writer.emitPackage(Constants.REALM_PACKAGE_NAME)
                .emitEmptyLine();

        ArrayList<String> imports = new ArrayList<String>();
        imports.add("android.util.JsonReader");
        imports.add("android.util.JsonToken");
        imports.add("io.realm.RealmFieldType");
        imports.add("io.realm.exceptions.RealmMigrationNeededException");
        imports.add("io.realm.internal.ColumnInfo");
        imports.add("io.realm.internal.RealmObjectProxy");
        imports.add("io.realm.internal.Table");
        imports.add("io.realm.internal.TableOrView");
        imports.add("io.realm.internal.ImplicitTransaction");
        imports.add("io.realm.internal.LinkView");
        imports.add("io.realm.internal.android.JsonUtils");
        imports.add("java.io.IOException");
        imports.add("java.util.ArrayList");
        imports.add("java.util.Collections");
        imports.add("java.util.List");
        imports.add("java.util.Iterator");
        imports.add("java.util.Date");
        imports.add("java.util.Map");
        imports.add("java.util.HashMap");
        imports.add("org.json.JSONObject");
        imports.add("org.json.JSONException");
        imports.add("org.json.JSONArray");

        Collections.sort(imports);
        writer.emitImports(imports);
        writer.emitEmptyLine();


        writer.beginType(
                qualifiedGeneratedClassName, 
                "class",                     
                EnumSet.of(Modifier.PUBLIC), 
                qualifiedClassName,          
                "RealmObjectProxy",          
                interfaceName)
                .emitEmptyLine();

        emitColumnIndicesClass(writer);

        emitClassFields(writer);
        emitConstructor(writer);
        emitAccessors(writer);
        emitInitTableMethod(writer);
        emitValidateTableMethod(writer);
        emitGetTableNameMethod(writer);
        emitGetFieldNamesMethod(writer);
        emitCreateOrUpdateUsingJsonObject(writer);
        emitCreateUsingJsonStream(writer);
        emitCopyOrUpdateMethod(writer);
        emitCopyMethod(writer);
        emitInsertMethod(writer);
        emitInsertListMethod(writer);
        emitInsertOrUpdateMethod(writer);
        emitInsertOrUpdateListMethod(writer);
        emitCreateDetachedCopyMethod(writer);
        emitUpdateMethod(writer);
        emitToStringMethod(writer);
        emitRealmObjectProxyImplementation(writer);
        emitHashcodeMethod(writer);
        emitEqualsMethod(writer);


        writer.endType();
        writer.close();
    }

    private void emitColumnIndicesClass(JavaWriter writer) throws IOException {
        writer.beginType(
                columnInfoClassName(),                       
                "class",                                     
                EnumSet.of(Modifier.STATIC, Modifier.FINAL), 
                "ColumnInfo")                                
                .emitEmptyLine();


        for (VariableElement variableElement : metadata.getFields()) {
            writer.emitField("long", columnIndexVarName(variableElement),
                    EnumSet.of(Modifier.PUBLIC, Modifier.FINAL));
        }
        writer.emitEmptyLine();


        writer.beginConstructor(EnumSet.noneOf(Modifier.class),
                "String", "path",
                "Table", "table");
        writer.emitStatement("final Map<String, Long> indicesMap = new HashMap<String, Long>(%s)",
                metadata.getFields().size());
        for (VariableElement variableElement : metadata.getFields()) {
            final String columnName = variableElement.getSimpleName().toString();
            final String columnIndexVarName = columnIndexVarName(variableElement);
            writer.emitStatement("this.%s = getValidColumnIndex(path, table, \"%s\", \"%s\")",
                    columnIndexVarName, simpleClassName, columnName);
            writer.emitStatement("indicesMap.put(\"%s\", this.%s)", columnName, columnIndexVarName);
            writer.emitEmptyLine();
        }
        writer.emitStatement("setIndicesMap(indicesMap)");
        writer.endConstructor();

        writer.endType();
        writer.emitEmptyLine();
    }

    private void emitClassFields(JavaWriter writer) throws IOException {
        writer.emitField(columnInfoClassName(), "columnInfo", EnumSet.of(Modifier.PRIVATE, Modifier.FINAL));
        writer.emitField("ProxyState", "proxyState", EnumSet.of(Modifier.PRIVATE, Modifier.FINAL));

        for (VariableElement variableElement : metadata.getFields()) {
            if (Utils.isRealmList(variableElement)) {
                String genericType = Utils.getGenericTypeQualifiedName(variableElement);
                writer.emitField("RealmList<" + genericType + ">", variableElement.getSimpleName().toString() + "RealmList", EnumSet.of(Modifier.PRIVATE));
            }
        }

        writer.emitField("List<String>", "FIELD_NAMES", EnumSet.of(Modifier.PRIVATE, Modifier.STATIC, Modifier.FINAL));
        writer.beginInitializer(true);
        writer.emitStatement("List<String> fieldNames = new ArrayList<String>()");
        for (VariableElement field : metadata.getFields()) {
            writer.emitStatement("fieldNames.add(\"%s\")", field.getSimpleName().toString());
        }
        writer.emitStatement("FIELD_NAMES = Collections.unmodifiableList(fieldNames)");
        writer.endInitializer();
        writer.emitEmptyLine();
    }

    private void emitConstructor(JavaWriter writer) throws IOException {

        writer.beginConstructor(EnumSet.noneOf(Modifier.class), "ColumnInfo", "columnInfo");
        writer.emitStatement("this.columnInfo = (%s) columnInfo", columnInfoClassName());
        writer.emitStatement("this.proxyState = new ProxyState(%s.class, this)", qualifiedClassName);
        writer.endConstructor();
        writer.emitEmptyLine();
    }

    private void emitAccessors(JavaWriter writer) throws IOException {
        for (VariableElement field : metadata.getFields()) {
            String fieldName = field.getSimpleName().toString();
            String fieldTypeCanonicalName = field.asType().toString();

            if (Constants.JAVA_TO_REALM_TYPES.containsKey(fieldTypeCanonicalName)) {
                
                String realmType = Constants.JAVA_TO_REALM_TYPES.get(fieldTypeCanonicalName);


                writer.emitAnnotation("SuppressWarnings", "\"cast\"");
                writer.beginMethod(fieldTypeCanonicalName, metadata.getGetter(fieldName), EnumSet.of(Modifier.PUBLIC));
                writer.emitStatement("proxyState.getRealm$realm().checkIfValid()");


                if (metadata.isNullable(field) && !Utils.isString(field) && !Utils.isByteArray(field)) {
                    writer.beginControlFlow("if (proxyState.getRow$realm().isNull(%s))", fieldIndexVariableReference(field));
                    writer.emitStatement("return null");
                    writer.endControlFlow();
                }


                String castingBackType;
                if (Utils.isBoxedType(field.asType().toString())) {
                    Types typeUtils = processingEnvironment.getTypeUtils();
                    castingBackType = typeUtils.unboxedType(field.asType()).toString();
                } else {
                    castingBackType = fieldTypeCanonicalName;
                }
                writer.emitStatement(
                        "return (%s) proxyState.getRow$realm().get%s(%s)",
                        castingBackType, realmType, fieldIndexVariableReference(field));
                writer.endMethod();
                writer.emitEmptyLine();


                writer.beginMethod("void", metadata.getSetter(fieldName), EnumSet.of(Modifier.PUBLIC), fieldTypeCanonicalName, "value");
                writer.emitStatement("proxyState.getRealm$realm().checkIfValid()");


                if (metadata.isNullable(field)) {
                    writer.beginControlFlow("if (value == null)")
                        .emitStatement("proxyState.getRow$realm().setNull(%s)", fieldIndexVariableReference(field))
                        .emitStatement("return")
                    .endControlFlow();
                } else if (!metadata.isNullable(field) && !Utils.isPrimitiveType(field)) {

                    writer
                        .beginControlFlow("if (value == null)")
                            .emitStatement(Constants.STATEMENT_EXCEPTION_ILLEGAL_NULL_VALUE, fieldName)
                        .endControlFlow();
                }
                writer.emitStatement(
                        "proxyState.getRow$realm().set%s(%s, value)",
                        realmType, fieldIndexVariableReference(field));
                writer.endMethod();
            } else if (Utils.isRealmModel(field)) {
                


                writer.beginMethod(fieldTypeCanonicalName, metadata.getGetter(fieldName), EnumSet.of(Modifier.PUBLIC));
                writer.emitStatement("proxyState.getRealm$realm().checkIfValid()");
                writer.beginControlFlow("if (proxyState.getRow$realm().isNullLink(%s))", fieldIndexVariableReference(field));
                        writer.emitStatement("return null");
                        writer.endControlFlow();
                writer.emitStatement("return proxyState.getRealm$realm().get(%s.class, proxyState.getRow$realm().getLink(%s))",
                        fieldTypeCanonicalName, fieldIndexVariableReference(field));
                writer.endMethod();
                writer.emitEmptyLine();


                writer.beginMethod("void", metadata.getSetter(fieldName), EnumSet.of(Modifier.PUBLIC), fieldTypeCanonicalName, "value");
                writer.emitStatement("proxyState.getRealm$realm().checkIfValid()");
                writer.beginControlFlow("if (value == null)");
                    writer.emitStatement("proxyState.getRow$realm().nullifyLink(%s)", fieldIndexVariableReference(field));
                    writer.emitStatement("return");
                writer.endControlFlow();
                writer.beginControlFlow("if (!RealmObject.isValid(value))");
                    writer.emitStatement("throw new IllegalArgumentException(\"'value' is not a valid managed object.\")");
                writer.endControlFlow();
                writer.beginControlFlow("if (((RealmObjectProxy)value).realmGet$proxyState().getRealm$realm() != proxyState.getRealm$realm())");
                    writer.emitStatement("throw new IllegalArgumentException(\"'value' belongs to a different Realm.\")");
                writer.endControlFlow();
                writer.emitStatement("proxyState.getRow$realm().setLink(%s, ((RealmObjectProxy)value).realmGet$proxyState().getRow$realm().getIndex())", fieldIndexVariableReference(field));
                writer.endMethod();
            } else if (Utils.isRealmList(field)) {
                
                String genericType = Utils.getGenericTypeQualifiedName(field);


                writer.beginMethod(fieldTypeCanonicalName, metadata.getGetter(fieldName), EnumSet.of(Modifier.PUBLIC));
                writer.emitStatement("proxyState.getRealm$realm().checkIfValid()");
                writer.emitSingleLineComment("use the cached value if available");
                writer.beginControlFlow("if (" + fieldName + "RealmList != null)");
                        writer.emitStatement("return " + fieldName + "RealmList");
                writer.nextControlFlow("else");
                    writer.emitStatement("LinkView linkView = proxyState.getRow$realm().getLinkList(%s)", fieldIndexVariableReference(field));
                    writer.emitStatement(fieldName + "RealmList = new RealmList<%s>(%s.class, linkView, proxyState.getRealm$realm())",
                        genericType, genericType);
                    writer.emitStatement("return " + fieldName + "RealmList");
                writer.endControlFlow();

                writer.endMethod();
                writer.emitEmptyLine();


                writer.beginMethod("void", metadata.getSetter(fieldName), EnumSet.of(Modifier.PUBLIC), fieldTypeCanonicalName, "value");
                writer.emitStatement("proxyState.getRealm$realm().checkIfValid()");
                writer.emitStatement("LinkView links = proxyState.getRow$realm().getLinkList(%s)", fieldIndexVariableReference(field));
                writer.emitStatement("links.clear()");
                writer.beginControlFlow("if (value == null)");
                    writer.emitStatement("return");
                writer.endControlFlow();
                writer.beginControlFlow("for (RealmModel linkedObject : (RealmList<? extends RealmModel>) value)");
                    writer.beginControlFlow("if (!RealmObject.isValid(linkedObject))");
                        writer.emitStatement("throw new IllegalArgumentException(\"Each element of 'value' must be a valid managed object.\")");
                    writer.endControlFlow();
                    writer.beginControlFlow("if (((RealmObjectProxy)linkedObject).realmGet$proxyState().getRealm$realm() != proxyState.getRealm$realm())");
                        writer.emitStatement("throw new IllegalArgumentException(\"Each element of 'value' must belong to the same Realm.\")");
                    writer.endControlFlow();
                    writer.emitStatement("links.add(((RealmObjectProxy)linkedObject).realmGet$proxyState().getRow$realm().getIndex())");
                writer.endControlFlow();
                writer.endMethod();
            } else {
                throw new UnsupportedOperationException(
                        String.format("Type '%s' of field '%s' is not supported", fieldTypeCanonicalName, fieldName));
            }
            writer.emitEmptyLine();
        }
    }

    private void emitRealmObjectProxyImplementation(JavaWriter writer) throws IOException {
        writer.emitAnnotation("Override");
        writer.beginMethod("ProxyState", "realmGet$proxyState", EnumSet.of(Modifier.PUBLIC));
        writer.emitStatement("return proxyState");
        writer.endMethod();
        writer.emitEmptyLine();
    }

    private void emitInitTableMethod(JavaWriter writer) throws IOException {
        writer.beginMethod(
                "Table", 
                "initTable", 
                EnumSet.of(Modifier.PUBLIC, Modifier.STATIC), 
                "ImplicitTransaction", "transaction"); 

        writer.beginControlFlow("if (!transaction.hasTable(\"" + Constants.TABLE_PREFIX + this.simpleClassName + "\"))");
        writer.emitStatement("Table table = transaction.getTable(\"%s%s\")", Constants.TABLE_PREFIX, this.simpleClassName);


        for (VariableElement field : metadata.getFields()) {
            String fieldName = field.getSimpleName().toString();
            String fieldTypeCanonicalName = field.asType().toString();
            String fieldTypeSimpleName = Utils.getFieldTypeSimpleName(field);

            if (Constants.JAVA_TO_REALM_TYPES.containsKey(fieldTypeCanonicalName)) {
                String nullableFlag;
                if (metadata.isNullable(field)) {
                    nullableFlag = "Table.NULLABLE";
                } else {
                    nullableFlag = "Table.NOT_NULLABLE";
                }
                writer.emitStatement("table.addColumn(%s, \"%s\", %s)",
                        Constants.JAVA_TO_COLUMN_TYPES.get(fieldTypeCanonicalName),
                        fieldName, nullableFlag);
            } else if (Utils.isRealmModel(field)) {
                writer.beginControlFlow("if (!transaction.hasTable(\"%s%s\"))", Constants.TABLE_PREFIX, fieldTypeSimpleName);
                writer.emitStatement("%s%s.initTable(transaction)", fieldTypeSimpleName, Constants.PROXY_SUFFIX);
                writer.endControlFlow();
                writer.emitStatement("table.addColumnLink(RealmFieldType.OBJECT, \"%s\", transaction.getTable(\"%s%s\"))",
                        fieldName, Constants.TABLE_PREFIX, fieldTypeSimpleName);
            } else if (Utils.isRealmList(field)) {
                String genericTypeSimpleName = Utils.getGenericTypeSimpleName(field);
                writer.beginControlFlow("if (!transaction.hasTable(\"%s%s\"))", Constants.TABLE_PREFIX, genericTypeSimpleName);
                writer.emitStatement("%s.initTable(transaction)", Utils.getProxyClassName(genericTypeSimpleName));
                writer.endControlFlow();
                writer.emitStatement("table.addColumnLink(RealmFieldType.LIST, \"%s\", transaction.getTable(\"%s%s\"))",
                        fieldName, Constants.TABLE_PREFIX, genericTypeSimpleName);
            }
        }

        for (VariableElement field : metadata.getIndexedFields()) {
            String fieldName = field.getSimpleName().toString();
            writer.emitStatement("table.addSearchIndex(table.getColumnIndex(\"%s\"))", fieldName);
        }

        if (metadata.hasPrimaryKey()) {
            String fieldName = metadata.getPrimaryKey().getSimpleName().toString();
            writer.emitStatement("table.setPrimaryKey(\"%s\")", fieldName);
        } else {
            writer.emitStatement("table.setPrimaryKey(\"\")");
        }

        writer.emitStatement("return table");
        writer.endControlFlow();
        writer.emitStatement("return transaction.getTable(\"%s%s\")", Constants.TABLE_PREFIX, this.simpleClassName);
        writer.endMethod();
        writer.emitEmptyLine();
    }

    private void emitValidateTableMethod(JavaWriter writer) throws IOException {
        writer.beginMethod(
                columnInfoClassName(), 
                "validateTable", 
                EnumSet.of(Modifier.PUBLIC, Modifier.STATIC), 
                "ImplicitTransaction", "transaction"); 

        writer.beginControlFlow("if (transaction.hasTable(\"" + Constants.TABLE_PREFIX + this.simpleClassName + "\"))");
        writer.emitStatement("Table table = transaction.getTable(\"%s%s\")", Constants.TABLE_PREFIX, this.simpleClassName);


        writer.beginControlFlow("if (table.getColumnCount() != " + metadata.getFields().size() + ")");
        writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath(), \"Field count does not match - expected %d but was \" + table.getColumnCount())",
                metadata.getFields().size());
        writer.endControlFlow();


        writer.emitStatement("Map<String, RealmFieldType> columnTypes = new HashMap<String, RealmFieldType>()");
        writer.beginControlFlow("for (long i = 0; i < " + metadata.getFields().size() + "; i++)");
        writer.emitStatement("columnTypes.put(table.getColumnName(i), table.getColumnType(i))");
        writer.endControlFlow();
        writer.emitEmptyLine();


        writer.emitStatement("final %1$s columnInfo = new %1$s(transaction.getPath(), table)", columnInfoClassName());
        writer.emitEmptyLine();


        long fieldIndex = 0;
        for (VariableElement field : metadata.getFields()) {
            String fieldName = field.getSimpleName().toString();
            String fieldTypeQualifiedName = Utils.getFieldTypeQualifiedName(field);
            String fieldTypeSimpleName = Utils.getFieldTypeSimpleName(field);

            if (Constants.JAVA_TO_REALM_TYPES.containsKey(fieldTypeQualifiedName)) {

                writer.beginControlFlow("if (!columnTypes.containsKey(\"%s\"))", fieldName);
                writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath(), \"Missing field '%s' in existing Realm file. " +
                        "Either remove field or migrate using io.realm.internal.Table.addColumn()." +
                        "\")", fieldName);
                writer.endControlFlow();
                writer.beginControlFlow("if (columnTypes.get(\"%s\") != %s)",
                        fieldName, Constants.JAVA_TO_COLUMN_TYPES.get(fieldTypeQualifiedName));
                writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath(), \"Invalid type '%s' for field '%s' in existing Realm file.\")",
                        fieldTypeSimpleName, fieldName);
                writer.endControlFlow();


                if (metadata.isNullable(field)) {
                    writer.beginControlFlow("if (!table.isColumnNullable(%s))", fieldIndexVariableReference(field));

                    if (field.equals(metadata.getPrimaryKey())) {
                        writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath()," +
                                "\"@PrimaryKey field '%s' does not support null values in the existing Realm file. " +
                                "Migrate using RealmObjectSchema.setNullable(), or mark the field as @Required.\")",
                                fieldName);

                    } else if (Utils.isBoxedType(fieldTypeQualifiedName)) {
                        writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath()," +
                                "\"Field '%s' does not support null values in the existing Realm file. " +
                                "Either set @Required, use the primitive type for field '%s' " +
                                "or migrate using RealmObjectSchema.setNullable().\")",
                                fieldName, fieldName);
                    } else {
                        writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath()," +
                                " \"Field '%s' is required. Either set @Required to field '%s' " +
                                "or migrate using RealmObjectSchema.setNullable().\")",
                                fieldName, fieldName);
                    }
                    writer.endControlFlow();
                } else {

                    if (field.equals(metadata.getPrimaryKey())) {
                        writer
                            .beginControlFlow("if (table.isColumnNullable(%s) && table.findFirstNull(%s) != TableOrView.NO_MATCH)",
                                    fieldIndexVariableReference(field), fieldIndexVariableReference(field))
                            .emitStatement("throw new IllegalStateException(\"Cannot migrate an object with null value in field '%s'." +
                                    " Either maintain the same type for primary key field '%s', or remove the object with null value before migration.\")",
                                    fieldName, fieldName)
                            .endControlFlow();
                    } else {
                        writer.beginControlFlow("if (table.isColumnNullable(%s))", fieldIndexVariableReference(field));
                        if (Utils.isPrimitiveType(fieldTypeQualifiedName)) {
                            writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath()," +
                                    " \"Field '%s' does support null values in the existing Realm file. " +
                                    "Use corresponding boxed type for field '%s' or migrate using RealmObjectSchema.setNullable().\")",
                                    fieldName, fieldName);
                        } else {
                            writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath()," +
                                    " \"Field '%s' does support null values in the existing Realm file. " +
                                    "Remove @Required or @PrimaryKey from field '%s' or migrate using RealmObjectSchema.setNullable().\")",
                                    fieldName, fieldName);
                        }
                        writer.endControlFlow();
                    }
                }


                if (field.equals(metadata.getPrimaryKey())) {
                    writer.beginControlFlow("if (table.getPrimaryKey() != table.getColumnIndex(\"%s\"))", fieldName);
                    writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath(), \"Primary key not defined for field '%s' in existing Realm file. Add @PrimaryKey.\")", fieldName);
                    writer.endControlFlow();
                }


                if (metadata.getIndexedFields().contains(field)) {
                    writer.beginControlFlow("if (!table.hasSearchIndex(table.getColumnIndex(\"%s\")))", fieldName);
                    writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath(), \"Index not defined for field '%s' in existing Realm file. " +
                            "Either set @Index or migrate using io.realm.internal.Table.removeSearchIndex().\")", fieldName);
                    writer.endControlFlow();
                }

            } else if (Utils.isRealmModel(field)) { 
                writer.beginControlFlow("if (!columnTypes.containsKey(\"%s\"))", fieldName);
                writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath(), \"Missing field '%s' in existing Realm file. " +
                        "Either remove field or migrate using io.realm.internal.Table.addColumn().\")", fieldName);
                writer.endControlFlow();
                writer.beginControlFlow("if (columnTypes.get(\"%s\") != RealmFieldType.OBJECT)", fieldName);
                writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath(), \"Invalid type '%s' for field '%s'\")",
                        fieldTypeSimpleName, fieldName);
                writer.endControlFlow();
                writer.beginControlFlow("if (!transaction.hasTable(\"%s%s\"))", Constants.TABLE_PREFIX, fieldTypeSimpleName);
                writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath(), \"Missing class '%s%s' for field '%s'\")",
                        Constants.TABLE_PREFIX, fieldTypeSimpleName, fieldName);
                writer.endControlFlow();

                writer.emitStatement("Table table_%d = transaction.getTable(\"%s%s\")", fieldIndex, Constants.TABLE_PREFIX, fieldTypeSimpleName);
                writer.beginControlFlow("if (!table.getLinkTarget(%s).hasSameSchema(table_%d))",
                        fieldIndexVariableReference(field), fieldIndex);
                writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath(), \"Invalid RealmObject for field '%s': '\" + table.getLinkTarget(%s).getName() + \"' expected - was '\" + table_%d.getName() + \"'\")",
                        fieldName, fieldIndexVariableReference(field), fieldIndex);
                writer.endControlFlow();
            } else if (Utils.isRealmList(field)) { 
                String genericTypeSimpleName = Utils.getGenericTypeSimpleName(field);
                writer.beginControlFlow("if (!columnTypes.containsKey(\"%s\"))", fieldName);
                writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath(), \"Missing field '%s'\")", fieldName);
                writer.endControlFlow();
                writer.beginControlFlow("if (columnTypes.get(\"%s\") != RealmFieldType.LIST)", fieldName);
                writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath(), \"Invalid type '%s' for field '%s'\")",
                        genericTypeSimpleName, fieldName);
                writer.endControlFlow();
                writer.beginControlFlow("if (!transaction.hasTable(\"%s%s\"))", Constants.TABLE_PREFIX, genericTypeSimpleName);
                writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath(), \"Missing class '%s%s' for field '%s'\")",
                        Constants.TABLE_PREFIX, genericTypeSimpleName, fieldName);
                writer.endControlFlow();

                writer.emitStatement("Table table_%d = transaction.getTable(\"%s%s\")", fieldIndex, Constants.TABLE_PREFIX, genericTypeSimpleName);
                writer.beginControlFlow("if (!table.getLinkTarget(%s).hasSameSchema(table_%d))",
                        fieldIndexVariableReference(field), fieldIndex);
                writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath(), \"Invalid RealmList type for field '%s': '\" + table.getLinkTarget(%s).getName() + \"' expected - was '\" + table_%d.getName() + \"'\")",
                        fieldName, fieldIndexVariableReference(field), fieldIndex);
                writer.endControlFlow();
            }
            fieldIndex++;
        }

        writer.emitStatement("return %s", "columnInfo");

        writer.nextControlFlow("else");
        writer.emitStatement("throw new RealmMigrationNeededException(transaction.getPath(), \"The '%s' class is missing from the schema for this Realm.\")", metadata.getSimpleClassName());
        writer.endControlFlow();
        writer.endMethod();
        writer.emitEmptyLine();
    }

    private void emitGetTableNameMethod(JavaWriter writer) throws IOException {
        writer.beginMethod("String", "getTableName", EnumSet.of(Modifier.PUBLIC, Modifier.STATIC));
        writer.emitStatement("return \"%s%s\"", Constants.TABLE_PREFIX, simpleClassName);
        writer.endMethod();
        writer.emitEmptyLine();
    }

    private void emitGetFieldNamesMethod(JavaWriter writer) throws IOException {
        writer.beginMethod("List<String>", "getFieldNames", EnumSet.of(Modifier.PUBLIC, Modifier.STATIC));
        writer.emitStatement("return FIELD_NAMES");
        writer.endMethod();
        writer.emitEmptyLine();
    }

    private void emitCopyOrUpdateMethod(JavaWriter writer) throws IOException {
        writer.beginMethod(
                qualifiedClassName, 
                "copyOrUpdate", 
                EnumSet.of(Modifier.PUBLIC, Modifier.STATIC), 
                "Realm", "realm", qualifiedClassName, "object", "boolean", "update", "Map<RealmModel,RealmObjectProxy>", "cache" 
        );

        writer
            .beginControlFlow("if (object instanceof RealmObjectProxy && ((RealmObjectProxy) object).realmGet$proxyState().getRealm$realm() != null && ((RealmObjectProxy) object).realmGet$proxyState().getRealm$realm().threadId != realm.threadId)")
                .emitStatement("throw new IllegalArgumentException(\"Objects which belong to Realm instances in other" +
                        " threads cannot be copied into this Realm instance.\")")
            .endControlFlow();


        writer
            .beginControlFlow("if (object instanceof RealmObjectProxy && ((RealmObjectProxy)object).realmGet$proxyState().getRealm$realm() != null && ((RealmObjectProxy)object).realmGet$proxyState().getRealm$realm().getPath().equals(realm.getPath()))")
                .emitStatement("return object")
            .endControlFlow();

        writer.emitStatement("RealmObjectProxy cachedRealmObject = cache.get(object)");
        writer.beginControlFlow("if (cachedRealmObject != null)")
                .emitStatement("return (%s) cachedRealmObject", qualifiedClassName)
                .nextControlFlow("else");

            if (!metadata.hasPrimaryKey()) {
                writer.emitStatement("return copy(realm, object, update, cache)");
            } else {
                writer
                    .emitStatement("%s realmObject = null", qualifiedClassName)
                    .emitStatement("boolean canUpdate = update")
                    .beginControlFlow("if (canUpdate)")
                        .emitStatement("Table table = realm.getTable(%s.class)", qualifiedClassName)
                        .emitStatement("long pkColumnIndex = table.getPrimaryKey()");

                String primaryKeyGetter = metadata.getPrimaryKeyGetter();
                VariableElement primaryKeyElement = metadata.getPrimaryKey();
                if (metadata.isNullable(primaryKeyElement)) {
                    if (Utils.isString(primaryKeyElement)) {
                        writer
                            .emitStatement("String value = ((%s) object).%s()", interfaceName, primaryKeyGetter)
                            .emitStatement("long rowIndex = TableOrView.NO_MATCH")
                            .beginControlFlow("if (value == null)")
                                .emitStatement("rowIndex = table.findFirstNull(pkColumnIndex)")
                            .nextControlFlow("else")
                                .emitStatement("rowIndex = table.findFirstString(pkColumnIndex, value)")
                            .endControlFlow();
                    } else {
                        writer
                            .emitStatement("Number value = ((%s) object).%s()", interfaceName, primaryKeyGetter)
                            .emitStatement("long rowIndex = TableOrView.NO_MATCH")
                            .beginControlFlow("if (value == null)")
                                .emitStatement("rowIndex = table.findFirstNull(pkColumnIndex)")
                            .nextControlFlow("else")
                                .emitStatement("rowIndex = table.findFirstLong(pkColumnIndex, value.longValue())")
                            .endControlFlow();
                    }
                } else {
                    String pkType = Utils.isString(metadata.getPrimaryKey()) ? "String" : "Long";
                    writer.emitStatement("long rowIndex = table.findFirst%s(pkColumnIndex, ((%s) object).%s())",
                            pkType, interfaceName, primaryKeyGetter);
                }

                writer
                    .beginControlFlow("if (rowIndex != TableOrView.NO_MATCH)")
                        .emitStatement("realmObject = new %s(realm.schema.getColumnInfo(%s.class))",
                                qualifiedGeneratedClassName,
                                qualifiedClassName)
                        .emitStatement("((RealmObjectProxy)realmObject).realmGet$proxyState().setRealm$realm(realm)")
                        .emitStatement("((RealmObjectProxy)realmObject).realmGet$proxyState().setRow$realm(table.getUncheckedRow(rowIndex))")
                        .emitStatement("cache.put(object, (RealmObjectProxy) realmObject)")
                    .nextControlFlow("else")
                        .emitStatement("canUpdate = false")
                    .endControlFlow();

                writer.endControlFlow();

                writer
                    .emitEmptyLine()
                    .beginControlFlow("if (canUpdate)")
                        .emitStatement("return update(realm, realmObject, object, cache)")
                    .nextControlFlow("else")
                        .emitStatement("return copy(realm, object, update, cache)")
                    .endControlFlow();
            }

        writer.endControlFlow();
        writer.endMethod();
        writer.emitEmptyLine();
    }

    private void setTableValues(JavaWriter writer, String fieldType, String fieldName, String interfaceName, String getter, boolean isUpdate) throws IOException {
        if ("long".equals(fieldType)
                || "int".equals(fieldType)
                || "short".equals(fieldType)
                || "byte".equals(fieldType)) {
            writer.emitStatement("Table.nativeSetLong(tableNativePtr, columnInfo.%sIndex, rowIndex, ((%s)object).%s())", fieldName, interfaceName, getter);

        } else if ("java.lang.Long".equals(fieldType)
                || "java.lang.Integer".equals(fieldType)
                || "java.lang.Short".equals(fieldType)
                || "java.lang.Byte".equals(fieldType)) {
            writer
                    .emitStatement("Number %s = ((%s)object).%s()", getter, interfaceName, getter)
                    .beginControlFlow("if (%s != null)", getter)
                        .emitStatement("Table.nativeSetLong(tableNativePtr, columnInfo.%sIndex, rowIndex, %s.longValue())", fieldName, getter);
                    if (isUpdate) {
                        writer.nextControlFlow("else")
                                .emitStatement("Table.nativeSetNull(tableNativePtr, columnInfo.%sIndex, rowIndex)", fieldName);
                    }
                    writer.endControlFlow();

        } else if ("double".equals(fieldType)) {
            writer.emitStatement("Table.nativeSetDouble(tableNativePtr, columnInfo.%sIndex, rowIndex, ((%s)object).%s())", fieldName, interfaceName, getter);

        } else if("java.lang.Double".equals(fieldType)) {
            writer
                    .emitStatement("Double %s = ((%s)object).%s()", getter, interfaceName, getter)
                    .beginControlFlow("if (%s != null)", getter)
                        .emitStatement("Table.nativeSetDouble(tableNativePtr, columnInfo.%sIndex, rowIndex, %s)", fieldName, getter);
                    if (isUpdate) {
                        writer.nextControlFlow("else")
                                .emitStatement("Table.nativeSetNull(tableNativePtr, columnInfo.%sIndex, rowIndex)", fieldName);
                    }
                    writer.endControlFlow();

        } else if ("float".equals(fieldType)) {
            writer.emitStatement("Table.nativeSetFloat(tableNativePtr, columnInfo.%sIndex, rowIndex, ((%s)object).%s())", fieldName, interfaceName, getter);

        } else if ("java.lang.Float".equals(fieldType)) {
            writer
                    .emitStatement("Float %s = ((%s)object).%s()", getter, interfaceName, getter)
                    .beginControlFlow("if (%s != null)", getter)
                        .emitStatement("Table.nativeSetFloat(tableNativePtr, columnInfo.%sIndex, rowIndex, %s)", fieldName, getter);
                    if (isUpdate) {
                        writer.nextControlFlow("else")
                                .emitStatement("Table.nativeSetNull(tableNativePtr, columnInfo.%sIndex, rowIndex)", fieldName);
                    }
                    writer.endControlFlow();

        } else if ("boolean".equals(fieldType)) {
            writer.emitStatement("Table.nativeSetBoolean(tableNativePtr, columnInfo.%sIndex, rowIndex, ((%s)object).%s())", fieldName, interfaceName, getter);

        } else if ("java.lang.Boolean".equals(fieldType)) {
            writer
                    .emitStatement("Boolean %s = ((%s)object).%s()", getter, interfaceName, getter)
                    .beginControlFlow("if (%s != null)", getter)
                        .emitStatement("Table.nativeSetBoolean(tableNativePtr, columnInfo.%sIndex, rowIndex, %s)", fieldName, getter);
                    if (isUpdate) {
                        writer.nextControlFlow("else")
                                .emitStatement("Table.nativeSetNull(tableNativePtr, columnInfo.%sIndex, rowIndex)", fieldName);
                    }
                    writer.endControlFlow();

        } else if ("byte[]".equals(fieldType)) {
            writer
                    .emitStatement("byte[] %s = ((%s)object).%s()", getter, interfaceName, getter)
                    .beginControlFlow("if (%s != null)", getter)
                        .emitStatement("Table.nativeSetByteArray(tableNativePtr, columnInfo.%sIndex, rowIndex, %s)", fieldName, getter);
                    if (isUpdate) {
                        writer.nextControlFlow("else")
                                .emitStatement("Table.nativeSetNull(tableNativePtr, columnInfo.%sIndex, rowIndex)", fieldName);
                    }
                    writer.endControlFlow();


        } else if ("java.util.Date".equals(fieldType)) {
            writer
                    .emitStatement("java.util.Date %s = ((%s)object).%s()", getter, interfaceName, getter)
                    .beginControlFlow("if (%s != null)", getter)
                        .emitStatement("Table.nativeSetTimestamp(tableNativePtr, columnInfo.%sIndex, rowIndex, %s.getTime())", fieldName, getter);
                    if (isUpdate) {
                        writer.nextControlFlow("else")
                                .emitStatement("Table.nativeSetNull(tableNativePtr, columnInfo.%sIndex, rowIndex)", fieldName);
                    }
                    writer.endControlFlow();

        } else if ("java.lang.String".equals(fieldType)) {
            writer
                    .emitStatement("String %s = ((%s)object).%s()", getter, interfaceName, getter)
                    .beginControlFlow("if (%s != null)", getter)
                        .emitStatement("Table.nativeSetString(tableNativePtr, columnInfo.%sIndex, rowIndex, %s)", fieldName, getter);
                    if (isUpdate) {
                        writer.nextControlFlow("else")
                                .emitStatement("Table.nativeSetNull(tableNativePtr, columnInfo.%sIndex, rowIndex)", fieldName);
                    }
                    writer.endControlFlow();
        } else {
            throw new IllegalStateException("Unsupported type " + fieldType);
        }
    }

    private void emitInsertMethod(JavaWriter writer) throws IOException {
        writer.beginMethod(
                "long", 
                "insert", 
                EnumSet.of(Modifier.PUBLIC, Modifier.STATIC), 
                "Realm", "realm", qualifiedClassName, "object", "Map<RealmModel,Long>", "cache" 
        );


        writer
                .beginControlFlow("if (object instanceof RealmObjectProxy && ((RealmObjectProxy)object).realmGet$proxyState().getRealm$realm() != null && ((RealmObjectProxy)object).realmGet$proxyState().getRealm$realm().getPath().equals(realm.getPath()))")
                .emitStatement("return ((RealmObjectProxy)object).realmGet$proxyState().getRow$realm().getIndex()")
                .endControlFlow();

        writer.emitStatement("Table table = realm.getTable(%s.class)", qualifiedClassName);
        writer.emitStatement("long tableNativePtr = table.getNativeTablePointer()");
        writer.emitStatement("%s columnInfo = (%s) realm.schema.getColumnInfo(%s.class)",
                columnInfoClassName(), columnInfoClassName(), qualifiedClassName);

        if (metadata.hasPrimaryKey()) {
            writer.emitStatement("long pkColumnIndex = table.getPrimaryKey()");
        }
        addPrimaryKeyCheckIfNeeeded(metadata, true, writer);

        for (VariableElement field : metadata.getFields()) {
            String fieldName = field.getSimpleName().toString();
            String fieldType = field.asType().toString();
            String getter = metadata.getGetter(fieldName);

            if (Utils.isRealmModel(field)) {
                writer
                        .emitEmptyLine()
                        .emitStatement("%s %sObj = ((%s) object).%s()", fieldType, fieldName, interfaceName, getter)
                        .beginControlFlow("if (%sObj != null)", fieldName)
                            .emitStatement("Long cache%1$s = cache.get(%1$sObj)", fieldName)
                            .beginControlFlow("if (cache%s == null)", fieldName)
                                .emitStatement("cache%s = %s.insert(realm, %sObj, cache)",
                                        fieldName,
                                        Utils.getProxyClassSimpleName(field),
                                        fieldName)
                            .endControlFlow()
                           .emitStatement("Table.nativeSetLink(tableNativePtr, columnInfo.%1$sIndex, rowIndex, cache%1$s)", fieldName)
                        .endControlFlow();
            } else if (Utils.isRealmList(field)) {
                final String genericType = Utils.getGenericTypeQualifiedName(field);
                writer
                        .emitEmptyLine()
                        .emitStatement("RealmList<%s> %sList = ((%s) object).%s()",
                                genericType, fieldName, interfaceName, getter)
                        .beginControlFlow("if (%sList != null)", fieldName)
                            .emitStatement("long %1$sNativeLinkViewPtr = Table.nativeGetLinkView(tableNativePtr, columnInfo.%1$sIndex, rowIndex)", fieldName)
                            .beginControlFlow("for (%1$s %2$sItem : %2$sList)", genericType, fieldName)
                                .emitStatement("Long cacheItemIndex%1$s = cache.get(%1$sItem)", fieldName)
                             .beginControlFlow("if (cacheItemIndex%s == null)", fieldName)
                                .emitStatement("cacheItemIndex%1$s = %2$s.insert(realm, %1$sItem, cache)", fieldName, Utils.getProxyClassSimpleName(field))
                             .endControlFlow()
                             .emitStatement("LinkView.nativeAdd(%1$sNativeLinkViewPtr, cacheItemIndex%1$s)", fieldName)
                            .endControlFlow()
                            .emitStatement("LinkView.nativeClose(%sNativeLinkViewPtr)", fieldName)
                        .endControlFlow()
                        .emitEmptyLine();

            } else {
                if (metadata.getPrimaryKey() != field) {
                    setTableValues(writer, fieldType, fieldName, interfaceName, getter, false);
                }
            }
        }

        writer.emitStatement("return rowIndex");
        writer.endMethod();
        writer.emitEmptyLine();
    }

    private void emitInsertListMethod(JavaWriter writer) throws IOException {
        writer.beginMethod(
                "void", 
                "insert", 
                EnumSet.of(Modifier.PUBLIC, Modifier.STATIC), 
                "Realm", "realm", "Iterator<? extends RealmModel>", "objects", "Map<RealmModel,Long>", "cache" 
        );

        writer.emitStatement("Table table = realm.getTable(%s.class)", qualifiedClassName);
        writer.emitStatement("long tableNativePtr = table.getNativeTablePointer()");
        writer.emitStatement("%s columnInfo = (%s) realm.schema.getColumnInfo(%s.class)",
                columnInfoClassName(), columnInfoClassName(), qualifiedClassName);
        if (metadata.hasPrimaryKey()) {
            writer.emitStatement("long pkColumnIndex = table.getPrimaryKey()");
        }
        writer.emitStatement("%s object = null", qualifiedClassName);

        writer.beginControlFlow("while (objects.hasNext())");
        writer.emitStatement("object = (%s) objects.next()", qualifiedClassName);
        writer.beginControlFlow("if(!cache.containsKey(object))");

        writer.beginControlFlow("if (object instanceof RealmObjectProxy && ((RealmObjectProxy)object).realmGet$proxyState().getRealm$realm() != null && ((RealmObjectProxy)object).realmGet$proxyState().getRealm$realm().getPath().equals(realm.getPath()))");
                writer.emitStatement("cache.put(object, ((RealmObjectProxy)object).realmGet$proxyState().getRow$realm().getIndex())");
                .nextControlFlow("else");
                addPrimaryKeyCheckIfNeeeded(metadata, true, writer);
                writer.endControlFlow();



        for (VariableElement field : metadata.getFields()) {
            String fieldName = field.getSimpleName().toString();
            String fieldType = field.asType().toString();
            String getter = metadata.getGetter(fieldName);

            if (Utils.isRealmModel(field)) {
                writer
                        .emitEmptyLine()
                        .emitStatement("%s %sObj = ((%s) object).%s()", fieldType, fieldName, interfaceName, getter)
                        .beginControlFlow("if (%sObj != null)", fieldName)
                            .emitStatement("Long cache%1$s = cache.get(%1$sObj)", fieldName)
                         .beginControlFlow("if (cache%s == null)", fieldName)
                                .emitStatement("cache%s = %s.insert(realm, %sObj, cache)",
                                        fieldName,
                                        Utils.getProxyClassSimpleName(field),
                                        fieldName)
                                .endControlFlow()
                        .emitStatement("table.setLink(columnInfo.%1$sIndex, rowIndex, cache%1$s)", fieldName)
                        .endControlFlow();
            } else if (Utils.isRealmList(field)) {
                final String genericType = Utils.getGenericTypeQualifiedName(field);
                writer
                        .emitEmptyLine()
                        .emitStatement("RealmList<%s> %sList = ((%s) object).%s()",
                                genericType, fieldName, interfaceName, getter)
                        .beginControlFlow("if (%sList != null)", fieldName)
                            .emitStatement("long %1$sNativeLinkViewPtr = Table.nativeGetLinkView(tableNativePtr, columnInfo.%1$sIndex, rowIndex)", fieldName)
                          .beginControlFlow("for (%1$s %2$sItem : %2$sList)", genericType, fieldName)
                                .emitStatement("Long cacheItemIndex%1$s = cache.get(%1$sItem)", fieldName)
                             .beginControlFlow("if (cacheItemIndex%s == null)", fieldName)
                                    .emitStatement("cacheItemIndex%1$s = %2$s.insert(realm, %1$sItem, cache)", fieldName, Utils.getProxyClassSimpleName(field))
                             .endControlFlow()
                        .emitStatement("LinkView.nativeAdd(%1$sNativeLinkViewPtr, cacheItemIndex%1$s)", fieldName)
                          .endControlFlow()
                        .emitStatement("LinkView.nativeClose(%sNativeLinkViewPtr)", fieldName)
                        .endControlFlow()
                        .emitEmptyLine();

            } else {
                if (metadata.getPrimaryKey() != field) {
                    setTableValues(writer, fieldType, fieldName, interfaceName, getter, false);
                }
            }
        }

        writer.endControlFlow();
        writer.endControlFlow();
        writer.endMethod();
        writer.emitEmptyLine();
    }

    private void emitInsertOrUpdateMethod(JavaWriter writer) throws IOException {
        writer.beginMethod(
                "long", 
                "insertOrUpdate", 
                EnumSet.of(Modifier.PUBLIC, Modifier.STATIC), 
                "Realm", "realm", qualifiedClassName, "object", "Map<RealmModel,Long>", "cache" 
        );


        writer
                .beginControlFlow("if (object instanceof RealmObjectProxy && ((RealmObjectProxy)object).realmGet$proxyState().getRealm$realm() != null && ((RealmObjectProxy)object).realmGet$proxyState().getRealm$realm().getPath().equals(realm.getPath()))")
                .emitStatement("return ((RealmObjectProxy)object).realmGet$proxyState().getRow$realm().getIndex()")
                .endControlFlow();

        writer.emitStatement("Table table = realm.getTable(%s.class)", qualifiedClassName);
        writer.emitStatement("long tableNativePtr = table.getNativeTablePointer()");
        writer.emitStatement("%s columnInfo = (%s) realm.schema.getColumnInfo(%s.class)",
                columnInfoClassName(), columnInfoClassName(), qualifiedClassName);

        if (metadata.hasPrimaryKey()) {
            writer.emitStatement("long pkColumnIndex = table.getPrimaryKey()");
        }
        addPrimaryKeyCheckIfNeeeded(metadata, false, writer);

        for (VariableElement field : metadata.getFields()) {
            String fieldName = field.getSimpleName().toString();
            String fieldType = field.asType().toString();
            String getter = metadata.getGetter(fieldName);

            if (Utils.isRealmModel(field)) {
                writer
                        .emitEmptyLine()
                        .emitStatement("%s %sObj = ((%s) object).%s()", fieldType, fieldName, interfaceName, getter)
                        .beginControlFlow("if (%sObj != null)", fieldName)
                            .emitStatement("Long cache%1$s = cache.get(%1$sObj)", fieldName)
                            .beginControlFlow("if (cache%s == null)", fieldName)
                                .emitStatement("cache%1$s = %2$s.insertOrUpdate(realm, %1$sObj, cache)",
                                        fieldName,
                                        Utils.getProxyClassSimpleName(field))
                            .endControlFlow()
                            .emitStatement("Table.nativeSetLink(tableNativePtr, columnInfo.%1$sIndex, rowIndex, cache%1$s)", fieldName)
                        .nextControlFlow("else")

                            .emitStatement("Table.nativeNullifyLink(tableNativePtr, columnInfo.%sIndex, rowIndex)", fieldName)
                        .endControlFlow();
            } else if (Utils.isRealmList(field)) {
                final String genericType = Utils.getGenericTypeQualifiedName(field);
                writer
                        .emitEmptyLine()
                        .emitStatement("long %1$sNativeLinkViewPtr = Table.nativeGetLinkView(tableNativePtr, columnInfo.%1$sIndex, rowIndex)", fieldName)
                        .emitStatement("LinkView.nativeClear(%sNativeLinkViewPtr)", fieldName)
                        .emitStatement("RealmList<%s> %sList = ((%s) object).%s()",
                                genericType, fieldName, interfaceName, getter)
                        .beginControlFlow("if (%sList != null)", fieldName)
                            .beginControlFlow("for (%1$s %2$sItem : %2$sList)", genericType, fieldName)
                                .emitStatement("Long cacheItemIndex%1$s = cache.get(%1$sItem)", fieldName)
                                .beginControlFlow("if (cacheItemIndex%s == null)", fieldName)
                                    .emitStatement("cacheItemIndex%1$s = %2$s.insertOrUpdate(realm, %1$sItem, cache)", fieldName, Utils.getProxyClassSimpleName(field))
                                .endControlFlow()
                                .emitStatement("LinkView.nativeAdd(%1$sNativeLinkViewPtr, cacheItemIndex%1$s)", fieldName)
                            .endControlFlow()
                        .endControlFlow()
                        .emitStatement("LinkView.nativeClose(%sNativeLinkViewPtr)", fieldName)
                        .emitEmptyLine();

            } else {
                if (metadata.getPrimaryKey() != field) {
                    setTableValues(writer, fieldType, fieldName, interfaceName, getter, true);
                }
            }
        }

        writer.emitStatement("return rowIndex");

        writer.endMethod();
        writer.emitEmptyLine();
    }

    private void emitInsertOrUpdateListMethod(JavaWriter writer) throws IOException {
        writer.beginMethod(
                "void", 
                "insertOrUpdate", 
                EnumSet.of(Modifier.PUBLIC, Modifier.STATIC), 
                "Realm", "realm", "Iterator<? extends RealmModel>", "objects", "Map<RealmModel,Long>", "cache" 
        );

        writer.emitStatement("Table table = realm.getTable(%s.class)", qualifiedClassName);
        writer.emitStatement("long tableNativePtr = table.getNativeTablePointer()");
        writer.emitStatement("%s columnInfo = (%s) realm.schema.getColumnInfo(%s.class)",
                columnInfoClassName(), columnInfoClassName(), qualifiedClassName);
        if (metadata.hasPrimaryKey()) {
            writer.emitStatement("long pkColumnIndex = table.getPrimaryKey()");
        }
        writer.emitStatement("%s object = null", qualifiedClassName);

        writer.beginControlFlow("while (objects.hasNext())");
        writer.emitStatement("object = (%s) objects.next()", qualifiedClassName);
        writer.beginControlFlow("if(!cache.containsKey(object))");

        writer
                .beginControlFlow("if (object instanceof RealmObjectProxy && ((RealmObjectProxy)object).realmGet$proxyState().getRealm$realm() != null && ((RealmObjectProxy)object).realmGet$proxyState().getRealm$realm().getPath().equals(realm.getPath()))")
        writer.emitStatement("cache.put(object, ((RealmObjectProxy)object).realmGet$proxyState().getRow$realm().getIndex())");
                .nextControlFlow("else");
        addPrimaryKeyCheckIfNeeeded(metadata, false, writer);
                writer.endControlFlow();

        for (VariableElement field : metadata.getFields()) {
            String fieldName = field.getSimpleName().toString();
            String fieldType = field.asType().toString();
            String getter = metadata.getGetter(fieldName);

            if (Utils.isRealmModel(field)) {
                writer
                        .emitEmptyLine()
                        .emitStatement("%s %sObj = ((%s) object).%s()", fieldType, fieldName, interfaceName, getter)
                        .beginControlFlow("if (%sObj != null)", fieldName)
                            .emitStatement("Long cache%1$s = cache.get(%1$sObj)", fieldName)
                            .beginControlFlow("if (cache%s == null)", fieldName)
                                .emitStatement("cache%1$s = %2$s.insertOrUpdate(realm, %1$sObj, cache)",
                                        fieldName,
                                        Utils.getProxyClassSimpleName(field))
                                    .endControlFlow()
                            .emitStatement("Table.nativeSetLink(tableNativePtr, columnInfo.%1$sIndex, rowIndex, cache%1$s)", fieldName)
                        .nextControlFlow("else")

                            .emitStatement("Table.nativeNullifyLink(tableNativePtr, columnInfo.%sIndex, rowIndex)", fieldName)
                        .endControlFlow();
            } else if (Utils.isRealmList(field)) {
                final String genericType = Utils.getGenericTypeQualifiedName(field);
                writer
                        .emitEmptyLine()
                        .emitStatement("long %1$sNativeLinkViewPtr = Table.nativeGetLinkView(tableNativePtr, columnInfo.%1$sIndex, rowIndex)", fieldName)
                        .emitStatement("LinkView.nativeClear(%sNativeLinkViewPtr)", fieldName)
                        .emitStatement("RealmList<%s> %sList = ((%s) object).%s()",
                                genericType, fieldName, interfaceName, getter)
                        .beginControlFlow("if (%sList != null)", fieldName)
                            .beginControlFlow("for (%1$s %2$sItem : %2$sList)", genericType, fieldName)
                                .emitStatement("Long cacheItemIndex%1$s = cache.get(%1$sItem)", fieldName)
                            .beginControlFlow("if (cacheItemIndex%s == null)", fieldName)
                                    .emitStatement("cacheItemIndex%1$s = %2$s.insertOrUpdate(realm, %1$sItem, cache)", fieldName, Utils.getProxyClassSimpleName(field))
                                .endControlFlow()
                            .emitStatement("LinkView.nativeAdd(%1$sNativeLinkViewPtr, cacheItemIndex%1$s)", fieldName)
                            .endControlFlow()
                        .endControlFlow()
                        .emitStatement("LinkView.nativeClose(%sNativeLinkViewPtr)", fieldName)
                        .emitEmptyLine();

            } else {
                if (metadata.getPrimaryKey() != field) {
                    setTableValues(writer, fieldType, fieldName, interfaceName, getter, true);
                }
            }
        }
            writer.endControlFlow();
        writer.endControlFlow();

        writer.endMethod();
        writer.emitEmptyLine();
    }

    private void addPrimaryKeyCheckIfNeeeded(ClassMetaData metadata, boolean throwIfPrimaryKeyDuplicate, JavaWriter writer) throws IOException {
        if (metadata.hasPrimaryKey()) {
            String primaryKeyGetter = metadata.getPrimaryKeyGetter();
            VariableElement primaryKeyElement = metadata.getPrimaryKey();
            if (metadata.isNullable(primaryKeyElement)) {
                if (Utils.isString(primaryKeyElement)) {
                    writer
                            .emitStatement("String primaryKeyValue = ((%s) object).%s()", interfaceName, primaryKeyGetter)
                            .emitStatement("long rowIndex = TableOrView.NO_MATCH")
                            .beginControlFlow("if (primaryKeyValue == null)")
                            .emitStatement("rowIndex = Table.nativeFindFirstNull(tableNativePtr, pkColumnIndex)")
                            .nextControlFlow("else")
                            .emitStatement("rowIndex = Table.nativeFindFirstString(tableNativePtr, pkColumnIndex, primaryKeyValue)")
                            .endControlFlow();
                } else {
                    writer
                            .emitStatement("Object primaryKeyValue = ((%s) object).%s()", interfaceName, primaryKeyGetter)
                            .emitStatement("long rowIndex = TableOrView.NO_MATCH")
                            .beginControlFlow("if (primaryKeyValue == null)")
                            .emitStatement("rowIndex = Table.nativeFindFirstNull(tableNativePtr, pkColumnIndex)")
                            .nextControlFlow("else")
                            .emitStatement("rowIndex = Table.nativeFindFirstInt(tableNativePtr, pkColumnIndex, ((%s) object).%s())", interfaceName, primaryKeyGetter)
                            .endControlFlow();
                }
            } else {
                writer.emitStatement("long rowIndex = TableOrView.NO_MATCH");
                writer.emitStatement("Object primaryKeyValue = ((%s) object).%s()", interfaceName, primaryKeyGetter);
                writer.beginControlFlow("if (primaryKeyValue != null)");

                if (Utils.isString(metadata.getPrimaryKey())) {
                    writer.emitStatement("rowIndex = Table.nativeFindFirstString(tableNativePtr, pkColumnIndex, (String)primaryKeyValue)");
                } else {
                    writer.emitStatement("rowIndex = Table.nativeFindFirstInt(tableNativePtr, pkColumnIndex, ((%s) object).%s())", interfaceName, primaryKeyGetter);
                }
                writer.endControlFlow();
            }

            writer.beginControlFlow("if (rowIndex == TableOrView.NO_MATCH)");
            writer.emitStatement("rowIndex = Table.nativeAddEmptyRow(tableNativePtr, 1)");
            if (Utils.isString(metadata.getPrimaryKey())) {
                writer.beginControlFlow("if (primaryKeyValue != null)");
                writer.emitStatement("Table.nativeSetString(tableNativePtr, pkColumnIndex, rowIndex, (String)primaryKeyValue)");
                writer.endControlFlow();
            } else {
                writer.beginControlFlow("if (primaryKeyValue != null)");
                writer.emitStatement("Table.nativeSetLong(tableNativePtr, pkColumnIndex, rowIndex, ((%s) object).%s())", interfaceName, primaryKeyGetter);
                writer.endControlFlow();
            }

            if (throwIfPrimaryKeyDuplicate) {
                writer.nextControlFlow("else");
                writer.emitStatement("Table.throwDuplicatePrimaryKeyException(primaryKeyValue)");
            }

            writer.endControlFlow();
            writer.emitStatement("cache.put(object, rowIndex)");
        } else {
            writer.emitStatement("long rowIndex = Table.nativeAddEmptyRow(tableNativePtr, 1)");
            writer.emitStatement("cache.put(object, rowIndex)");
        }
    }

    private void emitCopyMethod(JavaWriter writer) throws IOException {
        writer.beginMethod(
                qualifiedClassName, 
                "copy", 
                EnumSet.of(Modifier.PUBLIC, Modifier.STATIC), 
                "Realm", "realm", qualifiedClassName, "newObject", "boolean", "update", "Map<RealmModel,RealmObjectProxy>", "cache"); 

        writer.emitStatement("RealmObjectProxy cachedRealmObject = cache.get(newObject)");
        writer.beginControlFlow("if (cachedRealmObject != null)")
              .emitStatement("return (%s) cachedRealmObject", qualifiedClassName)
              .nextControlFlow("else");

            if (metadata.hasPrimaryKey()) {
                writer.emitStatement("%s realmObject = realm.createObject(%s.class, ((%s) newObject).%s())",
                        qualifiedClassName, qualifiedClassName, interfaceName, metadata.getPrimaryKeyGetter());
            } else {
                writer.emitStatement("%s realmObject = realm.createObject(%s.class)", qualifiedClassName, qualifiedClassName);
            }
            writer.emitStatement("cache.put(newObject, (RealmObjectProxy) realmObject)");
            for (VariableElement field : metadata.getFields()) {
                String fieldName = field.getSimpleName().toString();
                String fieldType = field.asType().toString();
                String setter = metadata.getSetter(fieldName);
                String getter = metadata.getGetter(fieldName);

                if (Utils.isRealmModel(field)) {
                    writer
                        .emitEmptyLine()
                        .emitStatement("%s %sObj = ((%s) newObject).%s()", fieldType, fieldName, interfaceName, getter)
                        .beginControlFlow("if (%sObj != null)", fieldName)
                            .emitStatement("%s cache%s = (%s) cache.get(%sObj)", fieldType, fieldName, fieldType, fieldName)
                            .beginControlFlow("if (cache%s != null)", fieldName)
                                .emitStatement("((%s) realmObject).%s(cache%s)", interfaceName, setter, fieldName)
                            .nextControlFlow("else")
                                .emitStatement("((%s) realmObject).%s(%s.copyOrUpdate(realm, %sObj, update, cache))",
                                        interfaceName,
                                        setter,
                                        Utils.getProxyClassSimpleName(field),
                                        fieldName)
                            .endControlFlow()
                        .nextControlFlow("else")

                            .emitStatement("((%s) realmObject).%s(null)", interfaceName, setter)
                        .endControlFlow();
                } else if (Utils.isRealmList(field)) {
                    final String genericType = Utils.getGenericTypeQualifiedName(field);
                    writer
                        .emitEmptyLine()
                        .emitStatement("RealmList<%s> %sList = ((%s) newObject).%s()",
                                genericType, fieldName, interfaceName, getter)
                        .beginControlFlow("if (%sList != null)", fieldName)
                            .emitStatement("RealmList<%s> %sRealmList = ((%s) realmObject).%s()",
                                    genericType, fieldName, interfaceName, getter)
                            .beginControlFlow("for (int i = 0; i < %sList.size(); i++)", fieldName)
                                    .emitStatement("%s %sItem = %sList.get(i)", genericType, fieldName, fieldName)
                                    .emitStatement("%s cache%s = (%s) cache.get(%sItem)", genericType, fieldName, genericType, fieldName)
                                    .beginControlFlow("if (cache%s != null)", fieldName)
                                            .emitStatement("%sRealmList.add(cache%s)", fieldName, fieldName)
                                    .nextControlFlow("else")
                                            .emitStatement("%sRealmList.add(%s.copyOrUpdate(realm, %sList.get(i), update, cache))", fieldName, Utils.getProxyClassSimpleName(field), fieldName)
                                    .endControlFlow()
                            .endControlFlow()
                        .endControlFlow()
                        .emitEmptyLine();

                } else {
                    writer.emitStatement("((%s) realmObject).%s(((%s) newObject).%s())",
                            interfaceName, setter, interfaceName, getter);
                }
            }

            writer.emitStatement("return realmObject");
          writer.endControlFlow();
        writer.endMethod();
        writer.emitEmptyLine();
    }

    private void emitCreateDetachedCopyMethod(JavaWriter writer) throws IOException {
        writer.beginMethod(
                qualifiedClassName, 
                "createDetachedCopy", 
                EnumSet.of(Modifier.PUBLIC, Modifier.STATIC), 
                qualifiedClassName, "realmObject", "int", "currentDepth", "int", "maxDepth", "Map<RealmModel, CacheData<RealmModel>>", "cache");
        writer
            .beginControlFlow("if (currentDepth > maxDepth || realmObject == null)")
                .emitStatement("return null")
            .endControlFlow()
            .emitStatement("CacheData<RealmModel> cachedObject = cache.get(realmObject)")
            .emitStatement("%s unmanagedObject", qualifiedClassName)
            .beginControlFlow("if (cachedObject != null)")
                .emitSingleLineComment("Reuse cached object or recreate it because it was encountered at a lower depth.")
                .beginControlFlow("if (currentDepth >= cachedObject.minDepth)")
                    .emitStatement("return (%s)cachedObject.object", qualifiedClassName)
                .nextControlFlow("else")
                    .emitStatement("unmanagedObject = (%s)cachedObject.object", qualifiedClassName)
                    .emitStatement("cachedObject.minDepth = currentDepth")
                .endControlFlow()
            .nextControlFlow("else")
                .emitStatement("unmanagedObject = new %s()", qualifiedClassName)
                .emitStatement("cache.put(realmObject, new RealmObjectProxy.CacheData(currentDepth, unmanagedObject))")
            .endControlFlow();

        for (VariableElement field : metadata.getFields()) {
            String fieldName = field.getSimpleName().toString();
            String setter = metadata.getSetter(fieldName);
            String getter = metadata.getGetter(fieldName);

            if (Utils.isRealmModel(field)) {
                writer
                    .emitEmptyLine()
                    .emitSingleLineComment("Deep copy of %s", fieldName)
                    .emitStatement("((%s) unmanagedObject).%s(%s.createDetachedCopy(((%s) realmObject).%s(), currentDepth + 1, maxDepth, cache))",
                                interfaceName, setter, Utils.getProxyClassSimpleName(field), interfaceName, getter);
            } else if (Utils.isRealmList(field)) {
                writer
                    .emitEmptyLine()
                    .emitSingleLineComment("Deep copy of %s", fieldName)
                    .beginControlFlow("if (currentDepth == maxDepth)")
                        .emitStatement("((%s) unmanagedObject).%s(null)", interfaceName, setter)
                    .nextControlFlow("else")
                        .emitStatement("RealmList<%s> managed%sList = ((%s) realmObject).%s()",
                                 Utils.getGenericTypeQualifiedName(field), fieldName, interfaceName, getter)
                        .emitStatement("RealmList<%1$s> unmanaged%2$sList = new RealmList<%1$s>()", Utils.getGenericTypeQualifiedName(field), fieldName)
                        .emitStatement("((%s) unmanagedObject).%s(unmanaged%sList)", interfaceName, setter, fieldName)
                        .emitStatement("int nextDepth = currentDepth + 1")
                        .emitStatement("int size = managed%sList.size()", fieldName)
                        .beginControlFlow("for (int i = 0; i < size; i++)")
                            .emitStatement("%s item = %s.createDetachedCopy(managed%sList.get(i), nextDepth, maxDepth, cache)",
                                    Utils.getGenericTypeQualifiedName(field), Utils.getProxyClassSimpleName(field), fieldName)
                            .emitStatement("unmanaged%sList.add(item)", fieldName)
                        .endControlFlow()
                    .endControlFlow();
            } else {
                writer.emitStatement("((%s) unmanagedObject).%s(((%s) realmObject).%s())",
                        interfaceName, setter, interfaceName, getter);
            }
        }

        writer.emitStatement("return unmanagedObject");
        writer.endMethod();
        writer.emitEmptyLine();
    }

    private void emitUpdateMethod(JavaWriter writer) throws IOException {
        if (!metadata.hasPrimaryKey()) {
            return;
        }

        writer.beginMethod(
                qualifiedClassName, 
                "update", 
                EnumSet.of(Modifier.STATIC), 
                "Realm", "realm", qualifiedClassName, "realmObject", qualifiedClassName, "newObject", "Map<RealmModel, RealmObjectProxy>", "cache"); 

        for (VariableElement field : metadata.getFields()) {
            String fieldName = field.getSimpleName().toString();
            String setter = metadata.getSetter(fieldName);
            String getter = metadata.getGetter(fieldName);
            if (Utils.isRealmModel(field)) {
                writer
                    .emitStatement("%s %sObj = ((%s) newObject).%s()",
                            Utils.getFieldTypeQualifiedName(field), fieldName, interfaceName, getter)
                    .beginControlFlow("if (%sObj != null)", fieldName)
                        .emitStatement("%s cache%s = (%s) cache.get(%sObj)", Utils.getFieldTypeQualifiedName(field), fieldName, Utils.getFieldTypeQualifiedName(field), fieldName)
                        .beginControlFlow("if (cache%s != null)", fieldName)
                            .emitStatement("((%s) realmObject).%s(cache%s)", interfaceName, setter, fieldName)
                        .nextControlFlow("else")
                            .emitStatement("((%s) realmObject).%s(%s.copyOrUpdate(realm, %sObj, true, cache))",
                                    interfaceName,
                                    setter,
                                    Utils.getProxyClassSimpleName(field),
                                    fieldName
                            )
                        .endControlFlow()
                    .nextControlFlow("else")

                        .emitStatement("((%s) realmObject).%s(null)", interfaceName, setter)
                    .endControlFlow();
            } else if (Utils.isRealmList(field)) {
                final String genericType = Utils.getGenericTypeQualifiedName(field);
                writer
                    .emitStatement("RealmList<%s> %sList = ((%s) newObject).%s()",
                            genericType, fieldName, interfaceName, getter)
                    .emitStatement("RealmList<%s> %sRealmList = ((%s) realmObject).%s()",
                            genericType, fieldName, interfaceName, getter)
                    .emitStatement("%sRealmList.clear()", fieldName)
                    .beginControlFlow("if (%sList != null)", fieldName)
                        .beginControlFlow("for (int i = 0; i < %sList.size(); i++)", fieldName)
                            .emitStatement("%s %sItem = %sList.get(i)", genericType, fieldName, fieldName)
                            .emitStatement("%s cache%s = (%s) cache.get(%sItem)", genericType, fieldName, genericType, fieldName)
                            .beginControlFlow("if (cache%s != null)", fieldName)
                                .emitStatement("%sRealmList.add(cache%s)", fieldName, fieldName)
                            .nextControlFlow("else")
                                .emitStatement("%sRealmList.add(%s.copyOrUpdate(realm, %sList.get(i), true, cache))", fieldName, Utils.getProxyClassSimpleName(field), fieldName)
                            .endControlFlow()
                        .endControlFlow()
                    .endControlFlow();

            } else {
                if (field == metadata.getPrimaryKey()) {
                    continue;
                }
                writer.emitStatement("((%s) realmObject).%s(((%s) newObject).%s())",
                        interfaceName, setter, interfaceName, getter);
            }
        }

        writer.emitStatement("return realmObject");
        writer.endMethod();
        writer.emitEmptyLine();
    }

    private void emitToStringMethod(JavaWriter writer) throws IOException {
        if (metadata.containsToString()) {
            return;
        }
        writer.emitAnnotation("Override");
        writer.beginMethod("String", "toString", EnumSet.of(Modifier.PUBLIC));
        writer.beginControlFlow("if (!RealmObject.isValid(this))");
        writer.emitStatement("return \"Invalid object\"");
        writer.endControlFlow();
        writer.emitStatement("StringBuilder stringBuilder = new StringBuilder(\"%s = [\")", simpleClassName);
        List<VariableElement> fields = metadata.getFields();
        for (int i = 0; i < fields.size(); i++) {
            VariableElement field = fields.get(i);
            String fieldName = field.getSimpleName().toString();

            writer.emitStatement("stringBuilder.append(\"{%s:\")", fieldName);
            if (Utils.isRealmModel(field)) {
                String fieldTypeSimpleName = Utils.getFieldTypeSimpleName(field);
                writer.emitStatement(
                        "stringBuilder.append(%s() != null ? \"%s\" : \"null\")",
                        metadata.getGetter(fieldName),
                        fieldTypeSimpleName
                );
            } else if (Utils.isRealmList(field)) {
                String genericTypeSimpleName = Utils.getGenericTypeSimpleName(field);
                writer.emitStatement("stringBuilder.append(\"RealmList<%s>[\").append(%s().size()).append(\"]\")",
                        genericTypeSimpleName,
                        metadata.getGetter(fieldName));
            } else {
                if (metadata.isNullable(field)) {
                    writer.emitStatement("stringBuilder.append(%s() != null ? %s() : \"null\")",
                            metadata.getGetter(fieldName),
                            metadata.getGetter(fieldName)
                    );
                } else {
                    writer.emitStatement("stringBuilder.append(%s())", metadata.getGetter(fieldName));
                }
            }
            writer.emitStatement("stringBuilder.append(\"}\")");

            if (i < fields.size() - 1) {
                writer.emitStatement("stringBuilder.append(\",\")");
            }
        }

        writer.emitStatement("stringBuilder.append(\"]\")");
        writer.emitStatement("return stringBuilder.toString()");
        writer.endMethod();
        writer.emitEmptyLine();
    }

    
    private void emitHashcodeMethod(JavaWriter writer) throws IOException {
        if (metadata.containsHashCode()) {
            return;
        }
        writer.emitAnnotation("Override");
        writer.beginMethod("int", "hashCode", EnumSet.of(Modifier.PUBLIC));
        writer.emitStatement("String realmName = proxyState.getRealm$realm().getPath()");
        writer.emitStatement("String tableName = proxyState.getRow$realm().getTable().getName()");
        writer.emitStatement("long rowIndex = proxyState.getRow$realm().getIndex()");
        writer.emitEmptyLine();
        writer.emitStatement("int result = 17");
        writer.emitStatement("result = 31 * result + ((realmName != null) ? realmName.hashCode() : 0)");
        writer.emitStatement("result = 31 * result + ((tableName != null) ? tableName.hashCode() : 0)");
        writer.emitStatement("result = 31 * result + (int) (rowIndex ^ (rowIndex >>> 32))");
        writer.emitStatement("return result");
        writer.endMethod();
        writer.emitEmptyLine();
    }

    private void emitEqualsMethod(JavaWriter writer) throws IOException {
        if (metadata.containsEquals()) {
            return;
        }
        String proxyClassName = Utils.getProxyClassName(simpleClassName);
        String otherObjectVarName = "a" + simpleClassName;
        writer.emitAnnotation("Override");
        writer.beginMethod("boolean", "equals", EnumSet.of(Modifier.PUBLIC), "Object", "o");
        writer.emitStatement("if (this == o) return true");
        writer.emitStatement("if (o == null || getClass() != o.getClass()) return false");
        writer.emitStatement("%s %s = (%s)o", proxyClassName, otherObjectVarName, proxyClassName);  
        writer.emitEmptyLine();
        writer.emitStatement("String path = proxyState.getRealm$realm().getPath()");
        writer.emitStatement("String otherPath = %s.proxyState.getRealm$realm().getPath()", otherObjectVarName);
        writer.emitStatement("if (path != null ? !path.equals(otherPath) : otherPath != null) return false;");
        writer.emitEmptyLine();
        writer.emitStatement("String tableName = proxyState.getRow$realm().getTable().getName()");
        writer.emitStatement("String otherTableName = %s.proxyState.getRow$realm().getTable().getName()", otherObjectVarName);
        writer.emitStatement("if (tableName != null ? !tableName.equals(otherTableName) : otherTableName != null) return false");
        writer.emitEmptyLine();
        writer.emitStatement("if (proxyState.getRow$realm().getIndex() != %s.proxyState.getRow$realm().getIndex()) return false", otherObjectVarName);
        writer.emitEmptyLine();
        writer.emitStatement("return true");
        writer.endMethod();
        writer.emitEmptyLine();
    }


    private void emitCreateOrUpdateUsingJsonObject(JavaWriter writer) throws IOException {
        writer.emitAnnotation("SuppressWarnings", "\"cast\"");
        writer.beginMethod(
                qualifiedClassName,
                "createOrUpdateUsingJsonObject",
                EnumSet.of(Modifier.PUBLIC, Modifier.STATIC),
                Arrays.asList("Realm", "realm", "JSONObject", "json", "boolean", "update"),
                Collections.singletonList("JSONException"));

        if (!metadata.hasPrimaryKey()) {
            writer.emitStatement("%s obj = realm.createObject(%s.class)", qualifiedClassName, qualifiedClassName);
        } else {
            String pkType = Utils.isString(metadata.getPrimaryKey()) ? "String" : "Long";
            writer
                .emitStatement("%s obj = null", qualifiedClassName)
                .beginControlFlow("if (update)")
                    .emitStatement("Table table = realm.getTable(%s.class)", qualifiedClassName)
                    .emitStatement("long pkColumnIndex = table.getPrimaryKey()")
                    .emitStatement("long rowIndex = TableOrView.NO_MATCH");
            if (metadata.isNullable(metadata.getPrimaryKey())) {
                writer
                    .beginControlFlow("if (json.isNull(\"%s\"))", metadata.getPrimaryKey().getSimpleName())
                        .emitStatement("rowIndex = table.findFirstNull(pkColumnIndex)")
                    .nextControlFlow("else")
                        .emitStatement("rowIndex = table.findFirst%s(pkColumnIndex, json.get%s(\"%s\"))",
                                pkType, pkType, metadata.getPrimaryKey().getSimpleName())
                    .endControlFlow();
            } else {
                writer
                    .beginControlFlow("if (!json.isNull(\"%s\"))", metadata.getPrimaryKey().getSimpleName())
                    .emitStatement("rowIndex = table.findFirst%s(pkColumnIndex, json.get%s(\"%s\"))",
                            pkType, pkType, metadata.getPrimaryKey().getSimpleName())
                    .endControlFlow();
            }
            writer
                    .beginControlFlow("if (rowIndex != TableOrView.NO_MATCH)")
                        .emitStatement("obj = new %s(realm.schema.getColumnInfo(%s.class))",
                                qualifiedGeneratedClassName, qualifiedClassName)
                        .emitStatement("((RealmObjectProxy)obj).realmGet$proxyState().setRealm$realm(realm)")
                        .emitStatement("((RealmObjectProxy)obj).realmGet$proxyState().setRow$realm(table.getUncheckedRow(rowIndex))")
                    .endControlFlow()
                .endControlFlow();

            writer.beginControlFlow("if (obj == null)");
            String primaryKeyFieldType = metadata.getPrimaryKey().asType().toString();
            String primaryKeyFieldName = metadata.getPrimaryKey().getSimpleName().toString();
            RealmJsonTypeHelper.emitCreateObjectWithPrimaryKeyValue(qualifiedClassName, qualifiedGeneratedClassName,
                    primaryKeyFieldType, primaryKeyFieldName, writer);
            writer.endControlFlow();
        }

        for (VariableElement field : metadata.getFields()) {
            String fieldName = field.getSimpleName().toString();
            String qualifiedFieldType = field.asType().toString();
            if (Utils.isRealmModel(field)) {
                RealmJsonTypeHelper.emitFillRealmObjectWithJsonValue(
                        interfaceName,
                        metadata.getSetter(fieldName),
                        fieldName,
                        qualifiedFieldType,
                        Utils.getProxyClassSimpleName(field),
                        writer
                );

            } else if (Utils.isRealmList(field)) {
                RealmJsonTypeHelper.emitFillRealmListWithJsonValue(
                        interfaceName,
                        metadata.getGetter(fieldName),
                        metadata.getSetter(fieldName),
                        fieldName,
                        ((DeclaredType) field.asType()).getTypeArguments().get(0).toString(),
                        Utils.getProxyClassSimpleName(field),
                        writer);

            } else {
                RealmJsonTypeHelper.emitFillJavaTypeWithJsonValue(
                        interfaceName,
                        metadata.getSetter(fieldName),
                        fieldName,
                        qualifiedFieldType,
                        writer
                );
            }
        }

        writer.emitStatement("return obj");
        writer.endMethod();
        writer.emitEmptyLine();
    }

    private void emitCreateUsingJsonStream(JavaWriter writer) throws IOException {
        writer.emitAnnotation("SuppressWarnings", "\"cast\"");
        writer.beginMethod(
                qualifiedClassName,
                "createUsingJsonStream",
                EnumSet.of(Modifier.PUBLIC, Modifier.STATIC),
                Arrays.asList("Realm", "realm", "JsonReader", "reader"),
                Collections.singletonList("IOException"));

        writer.emitStatement("%s obj = realm.createObject(%s.class)",qualifiedClassName, qualifiedClassName);
        writer.emitStatement("reader.beginObject()");
        writer.beginControlFlow("while (reader.hasNext())");
        writer.emitStatement("String name = reader.nextName()");

        List<VariableElement> fields = metadata.getFields();
        for (int i = 0; i < fields.size(); i++) {
            VariableElement field = fields.get(i);
            String fieldName = field.getSimpleName().toString();
            String qualifiedFieldType = field.asType().toString();

            if (i == 0) {
                writer.beginControlFlow("if (name.equals(\"%s\"))", fieldName);
            } else {
                writer.nextControlFlow("else if (name.equals(\"%s\"))", fieldName);
            }
            if (Utils.isRealmModel(field)) {
                RealmJsonTypeHelper.emitFillRealmObjectFromStream(
                        interfaceName,
                        metadata.getSetter(fieldName),
                        fieldName,
                        qualifiedFieldType,
                        Utils.getProxyClassSimpleName(field),
                        writer
                );

            } else if (Utils.isRealmList(field)) {
                RealmJsonTypeHelper.emitFillRealmListFromStream(
                        interfaceName,
                        metadata.getGetter(fieldName),
                        metadata.getSetter(fieldName),
                        ((DeclaredType) field.asType()).getTypeArguments().get(0).toString(),
                        Utils.getProxyClassSimpleName(field),
                        writer);

            } else {
                RealmJsonTypeHelper.emitFillJavaTypeFromStream(
                        interfaceName,
                        metadata.getSetter(fieldName),
                        fieldName,
                        qualifiedFieldType,
                        writer
                );
            }
        }

        if (fields.size() > 0) {
            writer.nextControlFlow("else");
            writer.emitStatement("reader.skipValue()");
            writer.endControlFlow();
        }
        writer.endControlFlow();
        writer.emitStatement("reader.endObject()");
        writer.emitStatement("return obj");
        writer.endMethod();
        writer.emitEmptyLine();
    }

    private String columnInfoClassName() {
        return simpleClassName + "ColumnInfo";
    }

    private String columnIndexVarName(VariableElement variableElement) {
        return variableElement.getSimpleName().toString() + "Index";
    }

    private String fieldIndexVariableReference(VariableElement variableElement) {
        return "columnInfo." + columnIndexVarName(variableElement);
    }
}

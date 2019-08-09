

package io.realm.internal;

import java.io.Closeable;
import java.util.Date;

import io.realm.exceptions.RealmException;



public class Table implements TableOrView, TableSchema, Closeable {

    public static final long INFINITE = -1;
    public static final String STRING_DEFAULT_VALUE = "";
    public static final long INTEGER_DEFAULT_VALUE = 0;

    private static final String PRIMARY_KEY_TABLE_NAME = "pk";
    private static final String PRIMARY_KEY_CLASS_COLUMN_NAME = "pk_table";
    private static final long PRIMARY_KEY_CLASS_COLUMN_INDEX = 0;
    private static final String PRIMARY_KEY_FIELD_COLUMN_NAME = "pk_property";
    private static final long PRIMARY_KEY_FIELD_COLUMN_INDEX = 1;
    private static final long NO_PRIMARY_KEY = -2;

    protected long nativePtr;
    
    protected final Object parent;
    private final Context context;
    private long cachedPrimaryKeyColumnIndex = NO_MATCH;


    protected int tableNo;
    protected boolean DEBUG = false;
    protected static int TableCount = 0;

    static {
        TightDB.loadLibrary();
    }


    
    public Table() {
        this.parent = null; 
        this.context = new Context();



        this.nativePtr = createNative();
        if (nativePtr == 0) {
            throw new java.lang.OutOfMemoryError("Out of native memory.");
        }
        if (DEBUG) {
            tableNo = ++TableCount;
            System.err.println("====== New Tablebase " + tableNo + " : ptr = " + nativePtr);
        }
    }

    protected native long createNative();
    
    Table(Context context, Object parent, long nativePointer) {
        this.context = context;
        this.parent  = parent;
        this.nativePtr = nativePointer;

        if (DEBUG) {
            tableNo = ++TableCount;
            System.err.println("===== New Tablebase(ptr) " + tableNo + " : ptr = " + nativePtr);
        }
    }

    @Override
    public Table getTable() {
        return this;
    }



    @Override
    public void close() {
        synchronized (context) {
            if (nativePtr != 0) {
                nativeClose(nativePtr);
                if (DEBUG) {
                    TableCount--;
                    System.err.println("==== CLOSE " + tableNo + " ptr= " + nativePtr + " remaining " + TableCount);
                }
                
                nativePtr = 0;
            }   
        }
    }

    protected static native void nativeClose(long nativeTablePtr);
    
    @Override
    protected void finalize() {
        synchronized (context) {
            if (nativePtr != 0) {
                boolean isRoot = (parent == null);
                context.asyncDisposeTable(nativePtr, isRoot);
                nativePtr = 0; 
            }
        }

        if (DEBUG) 
            System.err.println("==== FINALIZE " + tableNo + "...");
    }

    

    public boolean isValid() {
        if (nativePtr == 0)
            return false;
        return nativeIsValid(nativePtr);
    }

    protected native boolean nativeIsValid(long nativeTablePtr);

    @Override
    public boolean equals(Object other) {
        if (this == other) return true;
        if (other == null) return false;
        if (!(other instanceof Table)) return false; 

        Table otherTable = (Table) other;
        return nativeEquals(nativePtr, otherTable.nativePtr);
    }

    protected native boolean nativeEquals(long nativeTablePtr, long nativeTableToComparePtr);

    private void verifyColumnName(String name) {
        if (name.length() > 63) {
            throw new IllegalArgumentException("Column names are currently limited to max 63 characters.");
        }
    }

    @Override
    public TableSchema getSubtableSchema(long columnIndex) {
        if(nativeIsRootTable(nativePtr) == false) {
            throw new UnsupportedOperationException("This is a subtable. Can only be called on root table.");
        }

        long[] newPath = new long[1];
        newPath[0] = columnIndex;
        return new SubtableSchema(nativePtr, newPath);
    }

    protected native boolean nativeIsRootTable(long nativeTablePtr);

    
    @Override
    public long addColumn (ColumnType type, String name) {
        verifyColumnName(name);
        return nativeAddColumn(nativePtr, type.getValue(), name);
    }

    protected native long nativeAddColumn(long nativeTablePtr, int type, String name);

    
    public long addColumnLink (ColumnType type, String name, Table table) {
        verifyColumnName(name);
        return nativeAddColumnLink(nativePtr, type.getValue(), name, table.nativePtr);
    }

    protected native long nativeAddColumnLink(long nativeTablePtr, int type, String name, long targetTablePtr);

    
    @Override
    public void removeColumn(long columnIndex) {
        nativeRemoveColumn(nativePtr, columnIndex);
    }

    protected native void nativeRemoveColumn(long nativeTablePtr, long columnIndex);

    
    @Override
    public void renameColumn(long columnIndex, String newName) {
        verifyColumnName(newName);
        nativeRenameColumn(nativePtr, columnIndex, newName);
    }

    protected native void nativeRenameColumn(long nativeTablePtr, long columnIndex, String name);


    
    public void updateFromSpec(TableSpec tableSpec) {
        checkImmutable();
        nativeUpdateFromSpec(nativePtr, tableSpec);
    }

    protected native void nativeUpdateFromSpec(long nativeTablePtr, TableSpec tableSpec);



    
    @Override
    public long size() {
        return nativeSize(nativePtr);
    }

    protected native long nativeSize(long nativeTablePtr);

    
    @Override
    public boolean isEmpty() {
        return size() == 0;
    }

    
    @Override
    public void clear() {
        checkImmutable();
        nativeClear(nativePtr);
    }

    protected native void nativeClear(long nativeTablePtr);


    
    @Override
    public long getColumnCount() {
        return nativeGetColumnCount(nativePtr);
    }

    protected native long nativeGetColumnCount(long nativeTablePtr);


    public TableSpec getTableSpec(){
        return nativeGetTableSpec(nativePtr);
    }

    protected native TableSpec nativeGetTableSpec(long nativeTablePtr);

    
    @Override
    public String getColumnName(long columnIndex) {
        return nativeGetColumnName(nativePtr, columnIndex);
    }

    protected native String nativeGetColumnName(long nativeTablePtr, long columnIndex);

    
    @Override
    public long getColumnIndex(String columnName) {
        if (columnName == null) throw new IllegalArgumentException("Column name can not be null.");
        return nativeGetColumnIndex(nativePtr, columnName);
    }
    
    protected native long nativeGetColumnIndex(long nativeTablePtr, String columnName);


    
    @Override
    public ColumnType getColumnType(long columnIndex) {
        return ColumnType.fromNativeValue(nativeGetColumnType(nativePtr, columnIndex));
    }

    protected native int nativeGetColumnType(long nativeTablePtr, long columnIndex);

    
    @Override
    public void remove(long rowIndex) {
        checkImmutable();
        nativeRemove(nativePtr, rowIndex);
    }

    protected native void nativeRemove(long nativeTablePtr, long rowIndex);

    @Override
    public void removeLast() {
        checkImmutable();
        nativeRemoveLast(nativePtr);
    }

    protected native void nativeRemoveLast(long nativeTablePtr);

    public void moveLastOver(long rowIndex) {
        checkImmutable();
        nativeMoveLastOver(nativePtr, rowIndex);
    }

    protected native void nativeMoveLastOver(long nativeTablePtr, long rowIndex);


    public long addEmptyRow() {
        checkImmutable();
        if (hasPrimaryKey()) {
            long primaryKeyColumnIndex = getPrimaryKey();
            ColumnType type = getColumnType(primaryKeyColumnIndex);
            switch(type) {
                case STRING:
                    if (findFirstString(primaryKeyColumnIndex, STRING_DEFAULT_VALUE) != NO_MATCH) {
                        throwDuplicatePrimaryKeyException(STRING_DEFAULT_VALUE);
                    }
                    break;
                case INTEGER:
                    if (findFirstLong(primaryKeyColumnIndex, INTEGER_DEFAULT_VALUE) != NO_MATCH) {
                        throwDuplicatePrimaryKeyException(INTEGER_DEFAULT_VALUE);
                    };
                    break;

                default:
                    throw new RealmException("Cannot check for duplicate rows for unsupported primary key type: " + type);
            }
        }

        return nativeAddEmptyRow(nativePtr, 1);
    }

    public long addEmptyRows(long rows) {
        checkImmutable();
        if (rows < 1) throw new IllegalArgumentException("'rows' must be > 0.");
        if (hasPrimaryKey()) {
           if (rows > 1) throw new RealmException("Multiple empty rows cannot be created if a primary key is defined for the table.");
           return addEmptyRow();
        }
        return nativeAddEmptyRow(nativePtr, rows);
    }

    protected native long nativeAddEmptyRow(long nativeTablePtr, long rows);


    
    public long add(Object... values) {
        long rowIndex = size();
        addAt(rowIndex, values);
        return rowIndex;
    }


    
    public void addAt(long rowIndex, Object... values) {
        checkImmutable();


        long size = size();
        if (rowIndex > size) {
            throw new IllegalArgumentException("rowIndex " + String.valueOf(rowIndex) +
                    " must be <= table.size() " + String.valueOf(size) + ".");
        }


        int columns = (int)getColumnCount();
        if (columns != values.length) {
            throw new IllegalArgumentException("The number of value parameters (" +
                    String.valueOf(values.length) +
                    ") does not match the number of columns in the table (" +
                    String.valueOf(columns) + ").");
        }
        ColumnType colTypes[] = new ColumnType[columns];
        for (int columnIndex = 0; columnIndex < columns; columnIndex++) {
            Object value = values[columnIndex];
            ColumnType colType = getColumnType(columnIndex);
            colTypes[columnIndex] = colType;
            if (!colType.matchObject(value)) {

                String providedType;
                if (value == null)
                    providedType = "null";
                else
                    providedType = value.getClass().toString();

                throw new IllegalArgumentException("Invalid argument no " + String.valueOf(1 + columnIndex) +
                        ". Expected a value compatible with column type " + colType + ", but got " + providedType + ".");
            }
        }


        for (long columnIndex = 0; columnIndex < columns; columnIndex++) {
            Object value = values[(int)columnIndex];
            switch (colTypes[(int)columnIndex]) {
            case BOOLEAN:
                nativeInsertBoolean(nativePtr, columnIndex, rowIndex, (Boolean)value);
                break;
            case INTEGER:
                long intValue = ((Number) value).longValue();
                assertIntValueIsLegal(columnIndex, intValue);
                nativeInsertLong(nativePtr, columnIndex, rowIndex, intValue);
                break;
            case FLOAT:
                nativeInsertFloat(nativePtr, columnIndex, rowIndex, ((Float)value).floatValue());
                break;
            case DOUBLE:
                nativeInsertDouble(nativePtr, columnIndex, rowIndex, ((Double)value).doubleValue());
                break;
            case STRING:
                String stringValue = (String) value;
                assertStringValueIsLegal(columnIndex, stringValue);
                nativeInsertString(nativePtr, columnIndex, rowIndex, (String)value);
                break;
            case DATE:
                nativeInsertDate(nativePtr, columnIndex, rowIndex, ((Date)value).getTime()/1000);
                break;
            case MIXED:
                nativeInsertMixed(nativePtr, columnIndex, rowIndex, Mixed.mixedValue(value));
                break;
            case BINARY:
                nativeInsertByteArray(nativePtr, columnIndex, rowIndex, (byte[])value);
                break;
            case TABLE:
                nativeInsertSubtable(nativePtr, columnIndex, rowIndex);
                insertSubtableValues(rowIndex, columnIndex, value);
                break;
            default:
                throw new RuntimeException("Unexpected columnType: " + String.valueOf(colTypes[(int)columnIndex]));
            }
        }

        nativeInsertDone(nativePtr);

    }

    private boolean isPrimaryKeyColumn(long columnIndex) {
        return columnIndex == getPrimaryKey();
    }

    private void insertSubtableValues(long rowIndex, long columnIndex, Object value) {
        if (value != null) {

            Table subtable = getSubtableDuringInsert(columnIndex, rowIndex);
            int rows = ((Object[])value).length;
            for (int i=0; i<rows; ++i) {
                Object rowArr = ((Object[])value)[i];
                subtable.addAt(i, (Object[])rowArr);
            }
        }
    }



    public void insertLinkList(long columnIndex, long rowIndex) {
        nativeInsertLinkList(nativePtr, columnIndex, rowIndex);
        getInternalMethods().insertDone();
    }

    private native void nativeInsertLinkList(long nativePtr, long columnIndex, long rowIndex);

    
    public TableView getSortedView(long columnIndex, TableView.Order order){

        context.executeDelayedDisposal();
        long nativeViewPtr = nativeGetSortedView(nativePtr, columnIndex, (order == TableView.Order.ascending));
        try {
            return new TableView(this.context, this, nativeViewPtr);
        } catch (RuntimeException e) {
            TableView.nativeClose(nativeViewPtr);
            throw e;
        }
    }

    
    public TableView getSortedView(long columnIndex){

        context.executeDelayedDisposal();
        long nativeViewPtr = nativeGetSortedView(nativePtr, columnIndex, true);
        try {
            return new TableView(this.context, this, nativeViewPtr);
        } catch (RuntimeException e) {
            TableView.nativeClose(nativeViewPtr);
            throw e;
        }

    }

    protected native long nativeGetSortedView(long nativeTableViewPtr, long columnIndex, boolean ascending);



    
    public void set(long rowIndex, Object... values) {
        checkImmutable();


        long size = size();
        if (rowIndex >= size) {
            throw new IllegalArgumentException("rowIndex " + String.valueOf(rowIndex) +
                    " must be < table.size() " + String.valueOf(size) + ".");
        }


        int columns = (int)getColumnCount();
        if (columns != values.length) {
            throw new IllegalArgumentException("The number of value parameters (" +
                    String.valueOf(values.length) +
                    ") does not match the number of columns in the table (" +
                    String.valueOf(columns) + ").");
        }

        ColumnType colTypes[] = new ColumnType[columns];
        for (int columnIndex = 0; columnIndex < columns; columnIndex++) {
            Object value = values[columnIndex];
            ColumnType colType = getColumnType(columnIndex);
            colTypes[columnIndex] = colType;
            if (!colType.matchObject(value)) {
                throw new IllegalArgumentException("Invalid argument no " + String.valueOf(1 + columnIndex) +
                        ". Expected a value compatible with column type " + colType + ", but got " + value.getClass() + ".");
            }
        }


        
        remove(rowIndex);
        addAt(rowIndex, values);
    }


    private InternalMethods internal = new InternalMethods();


    public InternalMethods getInternalMethods(){
        return this.internal;
    }

    
    public long getPrimaryKey() {
        if (cachedPrimaryKeyColumnIndex > 0 || cachedPrimaryKeyColumnIndex == NO_PRIMARY_KEY) {
            return cachedPrimaryKeyColumnIndex;
        } else {
            Table pkTable = getPrimaryKeyTable();
            if (pkTable == null) return NO_PRIMARY_KEY; 
            long rowIndex = pkTable.findFirstString(PRIMARY_KEY_CLASS_COLUMN_INDEX, getName());
            if (rowIndex != NO_MATCH) {
                cachedPrimaryKeyColumnIndex = pkTable.getRow(rowIndex).getLong(PRIMARY_KEY_FIELD_COLUMN_INDEX);
            } else {
                cachedPrimaryKeyColumnIndex = NO_PRIMARY_KEY;
            }

            return cachedPrimaryKeyColumnIndex;
        }
    }

    
    public boolean isPrimaryKey(long columnIndex) {
        if (columnIndex < 0) return false;
        return columnIndex == getPrimaryKey();
    }

    
    public boolean hasPrimaryKey() {
        return getPrimaryKey() >= 0;
    }

    void assertStringValueIsLegal(long columnIndex, String value) {
        if (value == null) throw new IllegalArgumentException("Null String is not allowed.");
        if (isPrimaryKey(columnIndex)) {
            if (value.equals(STRING_DEFAULT_VALUE)) throwIllegalPrimaryKeyException(STRING_DEFAULT_VALUE);
            if (findFirstString(columnIndex, value) != TableOrView.NO_MATCH) throwDuplicatePrimaryKeyException(value);
        }
    }

    void assertIntValueIsLegal(long columnIndex, long value) {
        if (isPrimaryKeyColumn(columnIndex)) {
            if (value == INTEGER_DEFAULT_VALUE) throwIllegalPrimaryKeyException(INTEGER_DEFAULT_VALUE);
            if (findFirstLong(columnIndex, value) != TableOrView.NO_MATCH) throwDuplicatePrimaryKeyException(value);
        }
    }

    private void throwDuplicatePrimaryKeyException(Object value) {
        throw new RealmException("Primary key constraint broken. Value already exists: " + value);
    }

    private void throwIllegalPrimaryKeyException(Object value) {
        throw new RealmException("\"" + value +"\" not allowed as value in a field that is a primary key.");
    }

    private void throwInvalidPrimaryKeyColumn(long columnIndex, Object value) {
        throw new RealmException(String.format("Field \"%s\" cannot be a primary key, it already contains duplicate values: %s", getColumnName(columnIndex), value));
    }



    public class InternalMethods{

        public void insertLong(long columnIndex, long rowIndex, long value) {
            checkImmutable();
            nativeInsertLong(nativePtr, columnIndex, rowIndex, value);
        }

        public void insertDouble(long columnIndex, long rowIndex, double value) {
            checkImmutable();
            nativeInsertDouble(nativePtr, columnIndex, rowIndex, value);
        }

        public void insertFloat(long columnIndex, long rowIndex, float value) {
            checkImmutable();
            nativeInsertFloat(nativePtr, columnIndex, rowIndex, value);
        }

        public void insertBoolean(long columnIndex, long rowIndex, boolean value) {
            checkImmutable();
            nativeInsertBoolean(nativePtr, columnIndex, rowIndex, value);
        }

        public void insertDate(long columnIndex, long rowIndex, Date date) {
            checkImmutable();
            nativeInsertDate(nativePtr, columnIndex, rowIndex, date.getTime()/1000);
        }

        public void insertString(long columnIndex, long rowIndex, String value) {
            checkImmutable();
            nativeInsertString(nativePtr, columnIndex, rowIndex, value);
        }

        public void insertMixed(long columnIndex, long rowIndex, Mixed data) {
            checkImmutable();
            nativeInsertMixed(nativePtr, columnIndex, rowIndex, data);
        }

        

        public void insertBinary(long columnIndex, long rowIndex, byte[] data) {
            checkImmutable();
            if(data != null)
                nativeInsertByteArray(nativePtr, columnIndex, rowIndex, data);
            else
                throw new IllegalArgumentException("byte[] must not be null. Alternatively insert empty array.");
        }

        public void insertSubtable(long columnIndex, long rowIndex, Object[][] values) {
            checkImmutable();
            nativeInsertSubtable(nativePtr, columnIndex, rowIndex);
            insertSubtableValues(rowIndex, columnIndex, values);
        }

        public void insertDone() {
            checkImmutable();
            nativeInsertDone(nativePtr);
        }
    }

    protected native void nativeInsertFloat(long nativeTablePtr, long columnIndex, long rowIndex, float value);

    protected native void nativeInsertDouble(long nativeTablePtr, long columnIndex, long rowIndex, double value);

    protected native void nativeInsertLong(long nativeTablePtr, long columnIndex, long rowIndex, long value);

    protected native void nativeInsertBoolean(long nativeTablePtr, long columnIndex, long rowIndex, boolean value);

    protected native void nativeInsertDate(long nativePtr, long columnIndex, long rowIndex, long dateTimeValue);

    protected native void nativeInsertString(long nativeTablePtr, long columnIndex, long rowIndex, String value);

    protected native void nativeInsertMixed(long nativeTablePtr, long columnIndex, long rowIndex, Mixed mixed);


    

    protected native void nativeInsertByteArray(long nativePtr, long columnIndex, long rowIndex, byte[] data);

    protected native void nativeInsertSubtable(long nativeTablePtr, long columnIndex, long rowIndex);

    protected native void nativeInsertDone(long nativeTablePtr);





    @Override
    public long getLong(long columnIndex, long rowIndex) {
        return nativeGetLong(nativePtr, columnIndex, rowIndex);
    }

    protected native long nativeGetLong(long nativeTablePtr, long columnIndex, long rowIndex);

    @Override
    public boolean getBoolean(long columnIndex, long rowIndex) {
        return nativeGetBoolean(nativePtr, columnIndex, rowIndex);
    }

    protected native boolean nativeGetBoolean(long nativeTablePtr, long columnIndex, long rowIndex);

    @Override
    public float getFloat(long columnIndex, long rowIndex) {
        return nativeGetFloat(nativePtr, columnIndex, rowIndex);
    }

    protected native float nativeGetFloat(long nativeTablePtr, long columnIndex, long rowIndex);

    @Override
    public double getDouble(long columnIndex, long rowIndex) {
        return nativeGetDouble(nativePtr, columnIndex, rowIndex);
    }

    protected native double nativeGetDouble(long nativeTablePtr, long columnIndex, long rowIndex);

    @Override
    public Date getDate(long columnIndex, long rowIndex) {
        return new Date(nativeGetDateTime(nativePtr, columnIndex, rowIndex)*1000);
    }

    protected native long nativeGetDateTime(long nativeTablePtr, long columnIndex, long rowIndex);

    
    @Override
    public String getString(long columnIndex, long rowIndex) {
        return nativeGetString(nativePtr, columnIndex, rowIndex);
    }

    protected native String nativeGetString(long nativePtr, long columnIndex, long rowIndex);

    
    

    @Override
    public byte[] getBinaryByteArray(long columnIndex, long rowIndex) {
        return nativeGetByteArray(nativePtr, columnIndex, rowIndex);
    }

    protected native byte[] nativeGetByteArray(long nativePtr, long columnIndex, long rowIndex);

    @Override
    public Mixed getMixed(long columnIndex, long rowIndex) {
        return nativeGetMixed(nativePtr, columnIndex, rowIndex);
    }

    @Override
    public ColumnType getMixedType(long columnIndex, long rowIndex) {
        return ColumnType.fromNativeValue(nativeGetMixedType(nativePtr, columnIndex, rowIndex));
    }

    protected native int nativeGetMixedType(long nativePtr, long columnIndex, long rowIndex);

    protected native Mixed nativeGetMixed(long nativeTablePtr, long columnIndex, long rowIndex);

    public long getLink(long columnIndex, long rowIndex) {
        return nativeGetLink(nativePtr, columnIndex, rowIndex);
    }

    protected native long nativeGetLink(long nativePtr, long columnIndex, long rowIndex);


    public Table getLinkTarget(long columnIndex) {

        context.executeDelayedDisposal();
        long nativeTablePointer = nativeGetLinkTarget(nativePtr, columnIndex);
        try {

            return new Table(context, this.parent, nativeTablePointer);
        }
        catch (RuntimeException e) {
            Table.nativeClose(nativeTablePointer);
            throw e;
        }
    }

    protected native long nativeGetLinkTarget(long nativePtr, long columnIndex);


    
    @Override
    public Table getSubtable(long columnIndex, long rowIndex) {

        context.executeDelayedDisposal();
        long nativeSubtablePtr = nativeGetSubtable(nativePtr, columnIndex, rowIndex);
        try {

            return new Table(context, this, nativeSubtablePtr);
        }
        catch (RuntimeException e) {
            nativeClose(nativeSubtablePtr);
            throw e;
        }
    }

    protected native long nativeGetSubtable(long nativeTablePtr, long columnIndex, long rowIndex);




    private Table getSubtableDuringInsert(long columnIndex, long rowIndex) {

        context.executeDelayedDisposal();
        long nativeSubtablePtr =  nativeGetSubtableDuringInsert(nativePtr, columnIndex, rowIndex);
        try {
            return new Table(context, this, nativeSubtablePtr);
        }
        catch (RuntimeException e) {
            nativeClose(nativeSubtablePtr);
            throw e;
        }
    }

    private native long nativeGetSubtableDuringInsert(long nativeTablePtr, long columnIndex, long rowIndex);


    public long getSubtableSize(long columnIndex, long rowIndex) {
        return nativeGetSubtableSize(nativePtr, columnIndex, rowIndex);
    }

    protected native long nativeGetSubtableSize(long nativeTablePtr, long columnIndex, long rowIndex);

    public void clearSubtable(long columnIndex, long rowIndex) {
        checkImmutable();
        nativeClearSubtable(nativePtr, columnIndex, rowIndex);
    }

    protected native void nativeClearSubtable(long nativeTablePtr, long columnIndex, long rowIndex);


    public Row getRow(long index) {
        long nativeRowPtr = nativeGetRowPtr(nativePtr, index);
        return new Row(context, this, nativeRowPtr);
    }

    protected native long nativeGetRowPtr(long nativePtr, long index);






    @Override
    public void setLong(long columnIndex, long rowIndex, long value) {
        checkImmutable();
        assertIntValueIsLegal(columnIndex, value);
        nativeSetLong(nativePtr, columnIndex, rowIndex, value);
    }

    protected native void nativeSetLong(long nativeTablePtr, long columnIndex, long rowIndex, long value);

    @Override
    public void setBoolean(long columnIndex, long rowIndex, boolean value) {
        checkImmutable();
        nativeSetBoolean(nativePtr, columnIndex, rowIndex, value);
    }

    protected native void nativeSetBoolean(long nativeTablePtr, long columnIndex, long rowIndex, boolean value);

    @Override
    public void setFloat(long columnIndex, long rowIndex, float value) {
        checkImmutable();
        nativeSetFloat(nativePtr, columnIndex, rowIndex, value);
    }

    protected native void nativeSetFloat(long nativeTablePtr, long columnIndex, long rowIndex, float value);

    @Override
    public void setDouble(long columnIndex, long rowIndex, double value) {
        checkImmutable();
        nativeSetDouble(nativePtr, columnIndex, rowIndex, value);
    }

    protected native void nativeSetDouble(long nativeTablePtr, long columnIndex, long rowIndex, double value);

    @Override
    public void setDate(long columnIndex, long rowIndex, Date date) {
        if (date == null)
            throw new IllegalArgumentException("Null Date is not allowed.");
        checkImmutable();
        nativeSetDate(nativePtr, columnIndex, rowIndex, date.getTime() / 1000);
    }

    protected native void nativeSetDate(long nativeTablePtr, long columnIndex, long rowIndex, long dateTimeValue);

    @Override
    public void setString(long columnIndex, long rowIndex, String value) {
        checkImmutable();
        assertStringValueIsLegal(columnIndex, value);
        nativeSetString(nativePtr, columnIndex, rowIndex, value);
    }

    protected native void nativeSetString(long nativeTablePtr, long columnIndex, long rowIndex, String value);

    

    


    @Override
    public void setBinaryByteArray(long columnIndex, long rowIndex, byte[] data) {
        checkImmutable();
        if (data == null)
            throw new IllegalArgumentException("Null Array");
        nativeSetByteArray(nativePtr, columnIndex, rowIndex, data);
    }

    protected native void nativeSetByteArray(long nativePtr, long columnIndex, long rowIndex, byte[] data);

    
    @Override
    public void setMixed(long columnIndex, long rowIndex, Mixed data) {
        checkImmutable();
        if (data == null)
            throw new IllegalArgumentException();
        nativeSetMixed(nativePtr, columnIndex, rowIndex, data);
    }

    protected native void nativeSetMixed(long nativeTablePtr, long columnIndex, long rowIndex, Mixed data);

    public void setLink(long columnIndex, long rowIndex, long value) {
        checkImmutable();
        nativeSetLink(nativePtr, columnIndex, rowIndex, value);
    }

    protected native void nativeSetLink(long nativeTablePtr, long columnIndex, long rowIndex, long value);

    
    
    @Override
    public void adjust(long columnIndex, long value) {
        checkImmutable();
        nativeAddInt(nativePtr, columnIndex, value);
    }

    protected native void nativeAddInt(long nativeViewPtr, long columnIndex, long value);


    public void setIndex(long columnIndex) {
        checkImmutable();
        if (getColumnType(columnIndex) != ColumnType.STRING)
            throw new IllegalArgumentException("Index is only supported on string columns.");
        nativeSetIndex(nativePtr, columnIndex);
    }

    
    public void setPrimaryKey(String columnName) {
        Table pkTable = getPrimaryKeyTable();
        if (pkTable == null) throw new RealmException("Primary keys are only supported if Table is part of a Group");

        long rowIndex = pkTable.findFirstString(PRIMARY_KEY_CLASS_COLUMN_INDEX, getName());
        if (columnName == null || columnName.equals("")) {
            if (rowIndex > 0) pkTable.remove(rowIndex);
            cachedPrimaryKeyColumnIndex = NO_PRIMARY_KEY;
        } else {
            long primaryKeyColumnIndex = getColumnIndex(columnName);
            assertIsValidPrimaryKeyColumn(primaryKeyColumnIndex);
            if (rowIndex == NO_MATCH) {
                pkTable.add(getName(), primaryKeyColumnIndex);
            } else {
                pkTable.setLong(PRIMARY_KEY_FIELD_COLUMN_INDEX, rowIndex, primaryKeyColumnIndex);
            }

            cachedPrimaryKeyColumnIndex = primaryKeyColumnIndex;
        }
    }



    private void assertIsValidPrimaryKeyColumn(long columnIndex) {
        ColumnType columnType = getColumnType(columnIndex);
        TableView result = where().findAll();
        result.sort(columnIndex);

        switch (columnType) {
            case INTEGER:
                if (result.size() > 1) {
                    long value = result.getLong(columnIndex, 0);
                    for (long i = 1; i < result.size(); i++) {
                        long nextValue = result.getLong(columnIndex, i);
                        if (value == nextValue) {
                            throwInvalidPrimaryKeyColumn(columnIndex, value);
                        } else {
                            value = nextValue;
                        }
                    }
                }
                break;

            case STRING:
                if (result.size() > 1) {
                    String str = result.getString(columnIndex, 0);
                    for (int i = 1; i < result.size(); i++) {
                        String nextStr = result.getString(columnIndex, i);
                        if (str.equals(nextStr)) {
                            throwInvalidPrimaryKeyColumn(columnIndex, str);
                        } else {
                            str = nextStr;
                        }
                    }
                }
                break;

            default:
                throw new RealmException("Invalid primary key type: " + columnType);
        }
    }

    private Table getPrimaryKeyTable() {
        Group group = getTableGroup();
        if (group == null) return null;

        Table pkTable = group.getTable(PRIMARY_KEY_TABLE_NAME);
        if (pkTable.getColumnCount() == 0) {
            pkTable.addColumn(ColumnType.STRING, PRIMARY_KEY_CLASS_COLUMN_NAME);
            pkTable.addColumn(ColumnType.INTEGER, PRIMARY_KEY_FIELD_COLUMN_NAME);
        }

        return pkTable;
    }


    Group getTableGroup() {
        if (parent instanceof Group)  {
            return (Group) parent;
        } else if (parent instanceof Table) {
            return ((Table) parent).getTableGroup();
        } else {
            return null; 
        }
    }

    protected native void nativeSetIndex(long nativePtr, long columnIndex);


    public boolean hasIndex(long columnIndex) {
        return nativeHasIndex(nativePtr, columnIndex);
    }

    protected native boolean nativeHasIndex(long nativePtr, long columnIndex);


    public boolean isNullLink(long columnIndex, long rowIndex) {
        return nativeIsNullLink(nativePtr, columnIndex, rowIndex);
    }

    protected native boolean nativeIsNullLink(long nativePtr, long columnIndex, long rowIndex);

    public void nullifyLink(long columnIndex, long rowIndex) {
        nativeNullifyLink(nativePtr, columnIndex, rowIndex);
    }

    protected native void nativeNullifyLink(long nativePtr, long columnIndex, long rowIndex);


    boolean isImmutable() {
        if (!(parent instanceof Table)) {
            return parent != null && ((Group) parent).immutable;
        } else {
            return ((Table)parent).isImmutable();
        }
    }

    void checkImmutable() {
        if (isImmutable()) {
            throwImmutable();
        }
    }






    @Override
    public long sumLong(long columnIndex) {
        return nativeSumInt(nativePtr, columnIndex);
    }

    protected native long nativeSumInt(long nativePtr, long columnIndex);

    @Override
    public long maximumLong(long columnIndex) {
        return nativeMaximumInt(nativePtr, columnIndex);
    }

    protected native long nativeMaximumInt(long nativePtr, long columnIndex);

    @Override
    public long minimumLong(long columnIndex) {
        return nativeMinimumInt(nativePtr, columnIndex);
    }

    protected native long nativeMinimumInt(long nativePtr, long columnnIndex);

    @Override
    public double averageLong(long columnIndex) {
        return nativeAverageInt(nativePtr, columnIndex);
    }

    protected native double nativeAverageInt(long nativePtr, long columnIndex);


    @Override
    public double sumFloat(long columnIndex) {
        return nativeSumFloat(nativePtr, columnIndex);
    }

    protected native double nativeSumFloat(long nativePtr, long columnIndex);

    @Override
    public float maximumFloat(long columnIndex) {
        return nativeMaximumFloat(nativePtr, columnIndex);
    }

    protected native float nativeMaximumFloat(long nativePtr, long columnIndex);

    @Override
    public float minimumFloat(long columnIndex) {
        return nativeMinimumFloat(nativePtr, columnIndex);
    }

    protected native float nativeMinimumFloat(long nativePtr, long columnnIndex);

    @Override
    public double averageFloat(long columnIndex) {
        return nativeAverageFloat(nativePtr, columnIndex);
    }

    protected native double nativeAverageFloat(long nativePtr, long columnIndex);


    @Override
    public double sumDouble(long columnIndex) {
        return nativeSumDouble(nativePtr, columnIndex);
    }

    protected native double nativeSumDouble(long nativePtr, long columnIndex);

    @Override
    public double maximumDouble(long columnIndex) {
        return nativeMaximumDouble(nativePtr, columnIndex);
    }

    protected native double nativeMaximumDouble(long nativePtr, long columnIndex);

    @Override
    public double minimumDouble(long columnIndex) {
        return nativeMinimumDouble(nativePtr, columnIndex);
    }

    protected native double nativeMinimumDouble(long nativePtr, long columnnIndex);

    @Override
    public double averageDouble(long columnIndex) {
        return nativeAverageDouble(nativePtr, columnIndex);
    }

    protected native double nativeAverageDouble(long nativePtr, long columnIndex);



    @Override
    public Date maximumDate(long columnIndex) {
        return new Date(nativeMaximumDate(nativePtr, columnIndex) * 1000);
    }

    protected native long nativeMaximumDate(long nativePtr, long columnIndex);

    @Override
    public Date minimumDate(long columnIndex) {
        return new Date(nativeMinimumDate(nativePtr, columnIndex) * 1000);
    }

    protected native long nativeMinimumDate(long nativePtr, long columnnIndex);






    public long count(long columnIndex, long value) {
        return nativeCountLong(nativePtr, columnIndex, value);
    }

    protected native long nativeCountLong(long nativePtr, long columnIndex, long value);


    public long count(long columnIndex, float value) {
        return nativeCountFloat(nativePtr, columnIndex, value);
    }

    protected native long nativeCountFloat(long nativePtr, long columnIndex, float value);

    public long count(long columnIndex, double value) {
        return nativeCountDouble(nativePtr, columnIndex, value);
    }

    protected native long nativeCountDouble(long nativePtr, long columnIndex, double value);

    @Override
    public long count(long columnIndex, String value) {
        return nativeCountString(nativePtr, columnIndex, value);
    }

    protected native long nativeCountString(long nativePtr, long columnIndex, String value);






    @Override
    public TableQuery where() {

        context.executeDelayedDisposal();
        long nativeQueryPtr = nativeWhere(nativePtr);
        try {

            return new TableQuery(this.context, this, nativeQueryPtr);
        } catch (RuntimeException e) {
            TableQuery.nativeClose(nativeQueryPtr);
            throw e;
        }
    }

    protected native long nativeWhere(long nativeTablePtr);

    @Override
    public long findFirstLong(long columnIndex, long value) {
        return nativeFindFirstInt(nativePtr, columnIndex, value);
    }

    protected native long nativeFindFirstInt(long nativeTablePtr, long columnIndex, long value);

    @Override
    public long findFirstBoolean(long columnIndex, boolean value) {
        return nativeFindFirstBool(nativePtr, columnIndex, value);
    }

    protected native long nativeFindFirstBool(long nativePtr, long columnIndex, boolean value);

    @Override
    public long findFirstFloat(long columnIndex, float value) {
        return nativeFindFirstFloat(nativePtr, columnIndex, value);
    }

    protected native long nativeFindFirstFloat(long nativePtr, long columnIndex, float value);

    @Override
    public long findFirstDouble(long columnIndex, double value) {
        return nativeFindFirstDouble(nativePtr, columnIndex, value);
    }

    protected native long nativeFindFirstDouble(long nativePtr, long columnIndex, double value);

    @Override
    public long findFirstDate(long columnIndex, Date date) {
        return nativeFindFirstDate(nativePtr, columnIndex, date.getTime() / 1000);
    }

    protected native long nativeFindFirstDate(long nativeTablePtr, long columnIndex, long dateTimeValue);

    @Override
    public long findFirstString(long columnIndex, String value) {
        return nativeFindFirstString(nativePtr, columnIndex, value);
    }

    protected native long nativeFindFirstString(long nativeTablePtr, long columnIndex, String value);

    @Override
    public TableView findAllLong(long columnIndex, long value) {
        context.executeDelayedDisposal();
        long nativeViewPtr = nativeFindAllInt(nativePtr, columnIndex, value);
        try {
            return new TableView(this.context, this, nativeViewPtr);
        } catch (RuntimeException e) {
            TableView.nativeClose(nativeViewPtr);
            throw e;
        }
    }

    protected native long nativeFindAllInt(long nativePtr, long columnIndex, long value);

    @Override
    public TableView findAllBoolean(long columnIndex, boolean value) {

        context.executeDelayedDisposal();
        long nativeViewPtr = nativeFindAllBool(nativePtr, columnIndex, value);
        try {
            return new TableView(this.context, this, nativeViewPtr);
        } catch (RuntimeException e) {
            TableView.nativeClose(nativeViewPtr);
            throw e;
        }
    }

    protected native long nativeFindAllBool(long nativePtr, long columnIndex, boolean value);

    @Override
    public TableView findAllFloat(long columnIndex, float value) {

        context.executeDelayedDisposal();
        long nativeViewPtr = nativeFindAllFloat(nativePtr, columnIndex, value);
        try {
            return new TableView(this.context, this, nativeViewPtr);
        } catch (RuntimeException e) {
            TableView.nativeClose(nativeViewPtr);
            throw e;
        }
    }

    protected native long nativeFindAllFloat(long nativePtr, long columnIndex, float value);

    @Override
    public TableView findAllDouble(long columnIndex, double value) {

        context.executeDelayedDisposal();
        long nativeViewPtr = nativeFindAllDouble(nativePtr, columnIndex, value);
        try {
            return new TableView(this.context, this, nativeViewPtr);
        } catch (RuntimeException e) {
            TableView.nativeClose(nativeViewPtr);
            throw e;
        }
    }

    protected native long nativeFindAllDouble(long nativePtr, long columnIndex, double value);

    @Override
    public TableView findAllDate(long columnIndex, Date date) {

        context.executeDelayedDisposal();
        long nativeViewPtr = nativeFindAllDate(nativePtr, columnIndex, date.getTime() / 1000);
        try {
            return new TableView(this.context, this, nativeViewPtr);
        } catch (RuntimeException e) {
            TableView.nativeClose(nativeViewPtr);
            throw e;
        }
    }

    protected native long nativeFindAllDate(long nativePtr, long columnIndex, long dateTimeValue);

    @Override
    public TableView findAllString(long columnIndex, String value) {

        context.executeDelayedDisposal();
        long nativeViewPtr = nativeFindAllString(nativePtr, columnIndex, value);
        try {
            return new TableView(this.context, this, nativeViewPtr);
        } catch (RuntimeException e) {
            TableView.nativeClose(nativeViewPtr);
            throw e;
        }
    }

    protected native long nativeFindAllString(long nativePtr, long columnIndex, String value);


    @Override
    public long lowerBoundLong(long columnIndex, long value) {
        return nativeLowerBoundInt(nativePtr, columnIndex, value);
    }
    @Override
    public long upperBoundLong(long columnIndex, long value) {
        return nativeUpperBoundInt(nativePtr, columnIndex, value);
    }

    protected native long nativeLowerBoundInt(long nativePtr, long columnIndex, long value);
    protected native long nativeUpperBoundInt(long nativePtr, long columnIndex, long value);
    
    
    @Override
    public Table pivot(long stringCol, long intCol, PivotType pivotType){
        if (! this.getColumnType(stringCol).equals(ColumnType.STRING ))
            throw new UnsupportedOperationException("Group by column must be of type String");
        if (! this.getColumnType(intCol).equals(ColumnType.INTEGER ))
            throw new UnsupportedOperationException("Aggregation column must be of type Int");
        Table result = new Table();
        nativePivot(nativePtr, stringCol, intCol, pivotType.value, result.nativePtr);
        return result;
    }

    protected native void nativePivot(long nativeTablePtr, long stringCol, long intCol, int pivotType, long resultPtr);



    public TableView getDistinctView(long columnIndex) {

        this.context.executeDelayedDisposal();
        long nativeViewPtr = nativeGetDistinctView(nativePtr, columnIndex);
        try {
            return new TableView(this.context, this, nativeViewPtr);
        } catch (RuntimeException e) {
            TableView.nativeClose(nativeViewPtr);
            throw e;
        }
    }

    protected native long nativeGetDistinctView(long nativePtr, long columnIndex);
s
    
    public String getName() {
        return nativeGetName(nativePtr);
    }

    protected native String nativeGetName(long nativeTablePtr);


    public void optimize() {
        checkImmutable();
        nativeOptimize(nativePtr);
    }

    protected native void nativeOptimize(long nativeTablePtr);

    @Override
    public String toJson() {
        return nativeToJson(nativePtr);
    }

    protected native String nativeToJson(long nativeTablePtr);

    @Override
    public String toString() {
        return nativeToString(nativePtr, INFINITE);
    }

    @Override
    public String toString(long maxRows) {
        return nativeToString(nativePtr, maxRows);
    }

    protected native String nativeToString(long nativeTablePtr, long maxRows);

    @Override
    public String rowToString(long rowIndex) {
        return nativeRowToString(nativePtr, rowIndex);
    }

    protected native String nativeRowToString(long nativeTablePtr, long rowIndex);

    @Override
    public long sync() {
        throw new RuntimeException("Not supported for tables");
    }

    private void throwImmutable() {
        throw new IllegalStateException("Mutable method call during read transaction.");
    }
}

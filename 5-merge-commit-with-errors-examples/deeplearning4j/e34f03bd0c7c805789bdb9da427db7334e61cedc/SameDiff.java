package org.nd4j.autodiff.samediff;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.google.common.primitives.Ints;
import com.google.flatbuffers.FlatBufferBuilder;
import com.rits.cloning.Cloner;
import com.rits.cloning.IFastCloner;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.ArrayUtils;
import org.bytedeco.javacpp.BytePointer;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.functions.FunctionProperties;
import org.nd4j.autodiff.samediff.flow.FlowPath;
import org.nd4j.autodiff.util.cloner.DataBufferFastCloner;
import org.nd4j.autodiff.util.cloner.INDArrayFastCloner;
import org.nd4j.base.Preconditions;
import org.nd4j.graph.*;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.factory.DataBufferFactory;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.api.ops.impl.controlflow.If;
import org.nd4j.linalg.api.ops.impl.controlflow.While;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.*;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.*;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.GRUCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.SRU;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.SRUCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.GRUCellConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMCellConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.SRUCellConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.SRUConfiguration;
import org.nd4j.linalg.api.ops.impl.shape.Eye;
import org.nd4j.linalg.api.ops.impl.shape.tensorops.BaseTensorOp;
import org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArrayV3;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.GradientBackwardsMarker;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.collection.IntArrayKeyMap;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.exception.ND4UnresolvedOutputVariables;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.lossfunctions.impl.*;
import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.list.compat.TensorList;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.ConstantInitScheme;
import org.nd4j.weightinit.impl.NDArraySupplierInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import java.io.*;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;


@AllArgsConstructor
@Builder
@Slf4j
public class SameDiff {
    private Map<String, String[]> incomingArgsReverse;              
    private Map<String, String[]> outgoingArgsReverse;              
    private Map<String, int[]> permuteOrder;
    private boolean shouldBootStrap = true;
    private Set<String> importedVarName;


    private Map<String, String> baseNameForFunctionInstanceId;

    private DifferentialFunctionFactory functionFactory;
    private Map<String, SDVariable> variableMap;                    
    private Map<String, long[]> variableNameToShape;                

    private Map<String, SDVariable> gradients;                      
    private Map<String, SDVariable> forwardVarForGrad;

    private Map<String, INDArray> variableNameToArr;                


    private Map<String, List<DifferentialFunction>> functionsArgsFor;   
    private Map<String, List<DifferentialFunction>> functionOutputFor;  

    private Map<String, TensorList> lists = new HashMap<>();    


    private transient ThreadLocal<FlowPath> localFlowPath = new ThreadLocal<FlowPath>();


    private transient Map<String, Integer> reverseMap = null;


    
    private Map<String, List<String>> propertiesToResolve;

    
    private Map<String, Map<String, Object>> propertiesForFunction;


    private Map<String, List<String[]>> placeHolderMap;
    private Map<String, long[]> placeHolderOriginalShapes;
    private Set<String> placeHolderVarNames;
    private IdentityHashMap<INDArray, SDVariable> reverseArrayLookup;
    private MemoryWorkspace workspace;
    private Map<String, SameDiffFunctionDefinition> sameDiffFunctionDefinitionMap;
    private Map<String, SameDiff> sameDiffFunctionInstances;
    private Set<String> placeHolderFunctions;
    private static Cloner cloner = newCloner();
    private static Map<String, Method> opMethods;

    private Map<String, DifferentialFunction> functionInstancesById;

    private Table<String, String, String> fieldVariableResolutionMapping;


    private transient AtomicBoolean wasRegistered = new AtomicBoolean(false);



    @Getter
    private boolean debugMode;
    private Map<int[], Op> opsForResult;
    private boolean resolvedVariables = false;


    @Getter
    @Setter
    boolean logExecution = true;


    @Getter
    private SameDiff parent;

    @Getter
    private SameDiff child;


    static {
        opMethods = new HashMap<>();
        Method[] methods = SameDiff.class.getDeclaredMethods();
        for (Method method : methods) {
            if (method.getReturnType().equals(SDVariable.class)) {
                opMethods.put(method.getName(), method);
            }
        }
    }

    public static Cloner newCloner() {
        Cloner cloner = new Cloner();




        IFastCloner fc = new INDArrayFastCloner();
        cloner.registerFastCloner(Nd4j.getBackend().getNDArrayClass(), fc);
        cloner.registerFastCloner(Nd4j.getBackend().getComplexNDArrayClass(), fc);



        IFastCloner fc2 = new DataBufferFastCloner();
        DataBufferFactory d = Nd4j.getDataBufferFactory();
        doReg(cloner, fc2, d.intBufferClass());
        doReg(cloner, fc2, d.longBufferClass());
        doReg(cloner, fc2, d.halfBufferClass());
        doReg(cloner, fc2, d.floatBufferClass());
        doReg(cloner, fc2, d.doubleBufferClass());
        doReg(cloner, fc2, CompressedDataBuffer.class);
        return cloner;
    }

    private static void doReg(Cloner cl, IFastCloner fc, Class<?> c) {
        if (c != null)
            cl.registerFastCloner(c, fc);
    }


    
    public void updateVariableName(String varName, String withName) {
        SDVariable oldVarNameRef = getVariable(varName);
        variableMap.remove(oldVarNameRef.getVarName());
        val oldVarName = varName;
        oldVarNameRef.setVarName(withName);
        variableMap.put(withName, oldVarNameRef);


        for (val reverseValues : outgoingArgsReverse.entrySet()) {
            for (int i = 0; i < reverseValues.getValue().length; i++) {
                if (reverseValues.getValue()[i].equals(oldVarName)) {
                    reverseValues.getValue()[i] = withName;
                }
            }
        }


        for (val reverseValues : incomingArgsReverse.entrySet()) {
            for (int i = 0; i < reverseValues.getValue().length; i++) {
                if (reverseValues.getValue()[i].equals(oldVarName)) {
                    reverseValues.getValue()[i] = withName;
                }
            }
        }

        if (variableNameToArr.containsKey(oldVarName)) {
            val arr = variableNameToArr.remove(oldVarName);
            variableNameToArr.put(withName, arr);
        }


        if (variableNameToShape.containsKey(oldVarName)) {
            val shape = variableNameToShape.remove(oldVarName);
            variableNameToShape.put(withName, shape);
        }


        if (gradients.containsKey(oldVarName)) {
            val grad = gradients.remove(oldVarName);
            gradients.put(withName, grad);
        }

        if (forwardVarForGrad.containsKey(oldVarName)) {
            val forwardGrad = forwardVarForGrad.remove(oldVarName);
            forwardVarForGrad.put(withName, forwardGrad);
        }

        if (placeHolderMap.containsKey(oldVarName)) {
            val placeholders = placeHolderMap.remove(oldVarName);
            placeHolderMap.put(withName, placeholders);
        }


        if (functionsArgsFor.containsKey(oldVarName)) {
            val funcs = functionsArgsFor.remove(oldVarName);
            for (val func : funcs) {
                if (func instanceof BaseOp) {
                    BaseOp baseOp = (BaseOp) func;
                    if (baseOp.getXVertexId() != null && baseOp.getXVertexId().equals(oldVarName)) {
                        baseOp.setXVertexId(withName);
                    }

                    if (baseOp.getYVertexId() != null && baseOp.getYVertexId().equals(oldVarName)) {
                        baseOp.setYVertexId(withName);
                    }

                    if (baseOp.getZVertexId() != null && baseOp.getZVertexId().equals(oldVarName)) {
                        baseOp.setZVertexId(withName);
                    }

                }
            }

            functionsArgsFor.put(withName, funcs);
        }


        if (functionOutputFor.containsKey(oldVarName)) {
            val funcs = functionOutputFor.remove(oldVarName);
            for (val func : funcs) {
                if (func instanceof BaseOp) {
                    BaseOp baseOp = (BaseOp) func;
                    if (baseOp.getXVertexId() != null && baseOp.getXVertexId().equals(oldVarName)) {
                        baseOp.setXVertexId(withName);
                    }

                    if (baseOp.getYVertexId() != null && baseOp.getYVertexId().equals(oldVarName)) {
                        baseOp.setYVertexId(withName);
                    }

                    if (baseOp.getZVertexId() != null && baseOp.getZVertexId().equals(oldVarName)) {
                        baseOp.setZVertexId(withName);
                    }

                }
            }

            functionOutputFor.put(withName, funcs);
        }

        variableMap.remove(oldVarName);


    }


    
    public SameDiff disableDebugging() {
        debugMode = false;
        return this;
    }

    
    public SameDiff enableDebugMode() {
        debugMode = true;
        return this;
    }

    
    public DifferentialFunctionFactory f() {
        return functionFactory;
    }


    
    public SDVariable invokeGraphOn(SameDiff sameDiff) {

        Map<Integer, Integer> thisVertexIdToNew = new HashMap<>();
        int idx = 1;
        for (val var : variables()) {
            val clone = cloner.deepCloneDontCloneInstances(var, var.getSameDiff());
            val newVar = sameDiff.var(clone);
            if (var.getArr() != null) {
                sameDiff.associateArrayWithVariable(var.getArr(), newVar);
            }


            thisVertexIdToNew.put(idx, idx);
            clone.setSameDiff(sameDiff);
            idx++;

        }


        val newFunctions = new LinkedHashMap<String, DifferentialFunction>();
        for (DifferentialFunction function : functionInstancesById.values()) {
            if (function instanceof SDVariable) {
                continue;
            }

            DifferentialFunction clone = cloner.deepCloneDontCloneInstances(
                    function,
                    function.getSameDiff());
            clone.setSameDiff(sameDiff);
            clone.setOwnName(function.getOwnName());
            if (sameDiff.functionExists(function.getOwnName()))
                sameDiff.putFunctionForId(function.getOwnName(), function);
            newFunctions.put(function.getOwnName(), clone);

            val argsForFunction = function.args();
            val outputsForFunction = function.outputVariables();



            sameDiff.addArgsFor(argsForFunction, clone);
            sameDiff.addOutgoingFor(outputsForFunction, function);

            for (val arg : clone.args()) {
                arg.setSameDiff(sameDiff);
            }

            for (val output : clone.outputVariables()) {
                output.setSameDiff(sameDiff);
            }

            sameDiff.functionInstancesById.put(function.getOwnName(), function);
        }

        for (val reverseArrayEntry : reverseArrayLookup.entrySet()) {
            sameDiff.reverseArrayLookup.put(reverseArrayEntry.getKey(), sameDiff.getVariable(reverseArrayEntry.getValue().getVarName()));
        }

        return sameDiff.variables().get(sameDiff.variables().size() - 1);

    }


    
    public boolean functionExists(String id) {
        return functionInstancesById.containsKey(id);
    }


    
    public DifferentialFunction getFunctionById(String id) {
        if (!functionInstancesById.containsKey(id)) {
            throw new ND4JIllegalStateException("No function with id " + id + " found!");
        }
        return functionInstancesById.get(id);
    }


    
    public void putFunctionForId(String id, DifferentialFunction function) {
        if (functionInstancesById.containsKey(id)) {
            throw new ND4JIllegalStateException("Function by id already exists!");
        } else if (function instanceof SDVariable) {
            throw new ND4JIllegalStateException("Function must not be a variable!");
        }

        functionInstancesById.put(id, function);
    }


    
    public String[] getInputsForFunction(DifferentialFunction function) {
        if (!incomingArgsReverse.containsKey(function.getOwnName()))
            throw new ND4JIllegalStateException("Illegal function instance id found " + function.getOwnName());
        return incomingArgsReverse.get(function.getOwnName());
    }

    
    public String[] getOutputsForFunction(DifferentialFunction function) {
        return outgoingArgsReverse.get(function.getOwnName());
    }


    
    public SDVariable[] getOutputVariablesForFunction(DifferentialFunction function) {
        val inputs = getOutputsForFunction(function);
        if (inputs == null) {
            throw new ND4JIllegalStateException("No inputs found for function " + function);
        }

        val vars = new SDVariable[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            vars[i] = getVariable(inputs[i]);
        }

        return vars;
    }


    
    public SDVariable[] getInputVariablesForFunction(DifferentialFunction function) {
        val inputs = getInputsForFunction(function);
        if (inputs == null) {
            throw new ND4JIllegalStateException("No inputs found for function " + function);
        }

        val vars = new SDVariable[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            vars[i] = getVariable(inputs[i]);
            if (vars[i] == null) {
                throw new ND4JIllegalStateException("Found null variable at index " + i);
            }
        }

        return vars;
    }


    
    public void updateArrayForVarName(String varName, INDArray arr) {
        if (!variableNameToArr.containsKey(varName)) {
            throw new ND4JIllegalStateException("Array for " + varName + " does not exist. Please use putArrayForVertexId instead.");
        }

        variableNameToArr.put(varName, arr);
        reverseArrayLookup.put(arr, getVariable(varName));
    }

    
    public void putArrayForVarName(String varName, INDArray arr) {
        if (varName == null)
            throw new ND4JIllegalStateException("No null names allowed!");

        if (variableNameToArr.containsKey(varName)) {
            throw new ND4JIllegalStateException("Array for " + varName + " already exists!");
        }

        variableNameToArr.put(varName, arr);
    }


    
    public long[] getShapeForVarName(String varName) {
        if (variableNameToArr.containsKey(varName)) {
            return variableNameToArr.get(varName).shape();
        }

        return variableNameToShape.get(varName);
    }


    
    public void updateShapeForVarName(String varName, long[] shape) {
        if (shape == null) {
            throw new ND4JIllegalStateException("Null shapes not allowed!");
        }

        if (variableNameToArr.containsKey(varName) && !Arrays.equals(variableNameToArr.get(varName).shape(), shape)) {
            throw new ND4JIllegalStateException("Already found an existing array!");
        }


        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < 1) {
                addAsPlaceHolder(varName);
                placeHolderOriginalShapes.put(varName, shape);
                return;
            }
        }


        variableNameToShape.put(varName, shape);
    }


    
    public void putShapeForVarName(String varName, long[] shape) {
        if (shape == null) {
            throw new ND4JIllegalStateException("Shape must not be null!");
        }

        if (variableNameToShape.containsKey(varName)) {
            throw new ND4JIllegalStateException("Shape for " + varName + " already exists!");
        }

        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < 1) {
                addAsPlaceHolder(varName);
                placeHolderOriginalShapes.put(varName, shape);
                return;
            }
        }

        variableNameToShape.put(varName, shape);
    }


    
    public boolean shapeAlreadyExistsForVarName(String varName) {
        return variableNameToShape.containsKey(varName) || arrayAlreadyExistsForVarName(varName);
    }


    
    public boolean arrayAlreadyExistsForVarName(String varName) {
        return variableNameToArr.containsKey(varName);
    }

    
    public INDArray getArrForVarName(String varName) {
        return variableNameToArr.get(varName);
    }

    
    public void associateArrayWithVariable(INDArray arr, @NonNull String variable) {
        associateArrayWithVariable(arr, this.getVariable(variable));
    }

    
    public void associateArrayWithVariable(INDArray arr, SDVariable variable) {
        if (variable == null) {
            throw new ND4JIllegalArgumentException("Variable must not be null!");
        }

        if (arr == null) {
            throw new ND4JIllegalArgumentException("Array must not be null");
        }

        reverseArrayLookup.put(arr, variable);
        variableNameToArr.put(variable.getVarName(), arr);
        if (!shapeAlreadyExistsForVarName(variable.getVarName()))
            putShapeForVarName(variable.getVarName(), arr.shape());
        else {
            updateShapeForVarName(variable.getVarName(), arr.shape());
        }

        exec_cache = null;
    }


    
    public void putSubFunction(String name, SameDiff nameSpace) {
        if (sameDiffFunctionInstances.containsKey(name) && sameDiffFunctionInstances.get(name) != nameSpace) {
            throw new ND4JIllegalStateException("Unable to replace samediff namespace. Please choose another opName");
        }

        sameDiffFunctionInstances.put(name, nameSpace);
    }


    
    public Map<String, SDVariable> variableMap() {
        return variableMap;
    }


    
    public SDVariable invoke(Op op, SDVariable x, SDVariable y) {
        if (!opMethods.containsKey(op.opName())) {
            throw new ND4JIllegalStateException("Illegal method opName " + op.opName());
        }

        if (x != null && y != null) {
            try {
                return (SDVariable) opMethods.get(op.opName()).invoke(this, x, y);
            } catch (Exception e) {

            }
        } else {
            try {
                return (SDVariable) opMethods.get(op.opName()).invoke(this, x);
            } catch (Exception e) {

            }
        }

        throw new ND4JIllegalStateException("Illegal method opName " + op.opName());

    }


    
    public SDVariable getVariableForArray(INDArray arr) {
        return reverseArrayLookup.get(arr);
    }


    
    public Collection<String> definedFunctionNames() {
        return this.sameDiffFunctionInstances.keySet();
    }


    
    public long memoryForGraph() {
        return numElements() * DataTypeUtil.lengthForDtype(Nd4j.dataType());
    }

    
    public SDVariable invoke(Op op, SDVariable x) {
        return invoke(op, x, null);
    }

    private SameDiff() {
        functionFactory = new DifferentialFunctionFactory(this);
        variableMap = new LinkedHashMap<>();
        sameDiffFunctionDefinitionMap = new LinkedHashMap<>();
        sameDiffFunctionInstances = new LinkedHashMap<>();
        gradients = new LinkedHashMap<>();
        forwardVarForGrad = new LinkedHashMap<>();
        opsForResult = new IntArrayKeyMap<>();
        reverseArrayLookup = new IdentityHashMap<>();
        variableNameToArr = new LinkedHashMap<>();
        variableNameToShape = new LinkedHashMap<>();
        placeHolderMap = new LinkedHashMap<>();
        placeHolderVarNames = new LinkedHashSet<>();
        placeHolderOriginalShapes = new LinkedHashMap<>();
        incomingArgsReverse = new LinkedHashMap<>();
        outgoingArgsReverse = new LinkedHashMap<>();
        functionInstancesById = new LinkedHashMap<>();
        placeHolderFunctions = new LinkedHashSet<>();
        functionsArgsFor = new LinkedHashMap<>();
        functionOutputFor = new LinkedHashMap<>();
        baseNameForFunctionInstanceId = new LinkedHashMap<>();
        importedVarName = new LinkedHashSet<>();
        permuteOrder = new LinkedHashMap<>();
        propertiesToResolve = new LinkedHashMap<>();
        propertiesForFunction = new LinkedHashMap<>();
        fieldVariableResolutionMapping = HashBasedTable.create();

    }

    
    public void addPropertyToResolve(DifferentialFunction forFunction, String arrayName) {
        if (!propertiesToResolve.containsKey(forFunction.getOwnName())) {
            List<String> newVal = new ArrayList<>();
            newVal.add(arrayName);
            propertiesToResolve.put(forFunction.getOwnName(), newVal);
        } else {
            List<String> newVal = propertiesToResolve.get(forFunction.getOwnName());
            newVal.add(arrayName);
        }

    }

    
    public List<String> propertiesToResolveForFunction(DifferentialFunction function) {
        if (!propertiesToResolve.containsKey(function.getOwnName()))
            return Collections.emptyList();

        return propertiesToResolve.get(function.getOwnName());
    }


    
    public boolean hasPropertiesToResolve(DifferentialFunction function) {
        return propertiesToResolve.containsKey(function.getOwnName());
    }


    
    public <T> T getPropertyForFunction(DifferentialFunction functionInstance, String propertyName) {
        if (!propertiesForFunction.containsKey(functionInstance.getOwnName())) {
            return null;
        } else {
            val map = propertiesForFunction.get(functionInstance.getOwnName());
            return (T) map.get(propertyName);

        }
    }

    
    public void addPropertyForFunction(DifferentialFunction functionFor, String propertyName, INDArray property) {
        addPropertyForFunction(functionFor, propertyName, (Object) property);
    }


    
    public void addPropertyForFunction(DifferentialFunction functionFor, String propertyName, long property) {
        addPropertyForFunction(functionFor, propertyName, (Object) property);
    }


    private void addPropertyForFunction(DifferentialFunction functionFor, String propertyName, Object propertyValue) {
        if (!propertiesForFunction.containsKey(functionFor.getOwnName())) {
            Map<String, Object> fields = new LinkedHashMap<>();
            fields.put(propertyName, propertyValue);
            propertiesForFunction.put(functionFor.getOwnName(), fields);
        } else {
            val fieldMap = propertiesForFunction.get(functionFor.getOwnName());
            if (fieldMap.containsKey(propertyName)) {
                throw new ND4JIllegalStateException("Attempting to override property " + propertyName);
            }

            fieldMap.put(propertyName, propertyValue);
        }
    }


    
    public void addVariableMappingForField(DifferentialFunction function, String fieldName, String varName) {
        fieldVariableResolutionMapping.put(function.getOwnName(), fieldName, varName);
    }

    
    public String getVarNameForFieldAndFunction(DifferentialFunction function, String fieldName) {
        return fieldVariableResolutionMapping.get(function.getOwnName(), fieldName);
    }


    
    public boolean isImportVariable(String variableName) {
        return importedVarName.contains(variableName);
    }

    
    public void addVarNameForImport(String varName) {
        importedVarName.add(varName);
    }

    
    public void setBaseNameForFunctionInstanceId(String baseName, DifferentialFunction function) {
        baseNameForFunctionInstanceId.put(function.getOwnName(), baseName);
    }

    
    public String getBaseNameForFunction(DifferentialFunction function) {
        return baseNameForFunctionInstanceId.get(function.getOwnName());
    }


    
    public <X extends SDVariable> X setupFunction(X function) {
        Preconditions.checkNotNull(function, "Passed in function must not be null!");
        if (function instanceof SDVariable) {
            if (function.getSameDiff() != this) {
                function.setSameDiff(this);
            }
            return function;
        }
        return function;
    }


    
    public void addOutgoingFor(SDVariable[] variables, DifferentialFunction function) {
        String[] varNames = new String[variables.length];
        for (int i = 0; i < varNames.length; i++) {
            varNames[i] = variables[i].getVarName();
        }

        addOutgoingFor(varNames, function);
    }


    
    public void addOutgoingFor(String[] varNames, DifferentialFunction function) {

        if (function.getOwnName() == null)
            throw new ND4JIllegalStateException("Instance id can not be null. Function not initialized properly");

        if (outgoingArgsReverse.containsKey(function.getOwnName())) {
            throw new ND4JIllegalStateException("Outgoing arguments already declared for " + function);
        }

        if (varNames == null)
            throw new ND4JIllegalStateException("Var names can not be null!");


        for (int i = 0; i < varNames.length; i++) {
            if (varNames[i] == null)
                throw new ND4JIllegalStateException("Variable name elements can not be null!");
        }

        outgoingArgsReverse.put(function.getOwnName(), varNames);

        for (val resultName : varNames) {
            List<DifferentialFunction> funcs = functionOutputFor.get(resultName);
            if (funcs == null) {
                funcs = new ArrayList<>();
                functionOutputFor.put(resultName, funcs);
            }

            funcs.add(function);
        }

    }

    
    public void addArgsFor(String[] variables, DifferentialFunction function) {
        if (function.getOwnName() == null)
            throw new ND4JIllegalStateException("Instance id can not be null. Function not initialized properly");


        for (val varName : variables) {
            if (isPlaceHolder(varName)) {
                placeHolderFunctions.add(function.getOwnName());
            }
        }

        incomingArgsReverse.put(function.getOwnName(), variables);
        for (val variableName : variables) {
            List<DifferentialFunction> funcs = functionsArgsFor.get(variableName);
            if (funcs == null) {
                funcs = new ArrayList<>();
                functionsArgsFor.put(variableName, funcs);
            }

            funcs.add(function);
        }

    }


    
    public void addArgsFor(SDVariable[] variables, DifferentialFunction function) {
        String[] varNames = new String[variables.length];
        for (int i = 0; i < varNames.length; i++) {
            if (variables[i] == null)
                throw new ND4JIllegalStateException("Found null variable at index " + i);
            varNames[i] = variables[i].getVarName();
        }
        addArgsFor(varNames, function);
    }

    
    public DifferentialFunction getVariableOutputFunction(String variableName) {
        List<DifferentialFunction> list = functionOutputFor.get(variableName);
        if (list == null) {
            return null;
        }
        return list.get(0);
    }

    
    public List<DifferentialFunction> getVariableArgOfFunctions(String variableName) {
        return functionsArgsFor.get(variableName);
    }


    
    public boolean hasArgs(DifferentialFunction function) {
        String[] vertexIdArgs = incomingArgsReverse.get(function.getOwnName());
        return vertexIdArgs != null && vertexIdArgs.length > 0;
    }


    public DifferentialFunction[] functions() {
        val ret = functionInstancesById.values();
        return ret.toArray(new DifferentialFunction[ret.size()]);
    }


    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + (variableMap != null ? variableMap.hashCode() : 0);
        return result;
    }


    
    public static SameDiff create(SameDiff originalSameDiff) {
        SameDiff ret = SameDiff.builder()
                .variableMap(originalSameDiff.variableMap)
                .sameDiffFunctionInstances(originalSameDiff.sameDiffFunctionInstances)
                .build();

        DifferentialFunctionFactory differentialFunctionFactory =
                new
                        DifferentialFunctionFactory(ret);
        ret.functionFactory = differentialFunctionFactory;
        return ret;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        SameDiff sameDiff = (SameDiff) o;

        if (variableMap != null ? !variableMap.equals(sameDiff.variableMap) : sameDiff.variableMap != null)
            return false;
        if (sameDiffFunctionDefinitionMap != null ? !sameDiffFunctionDefinitionMap.equals(sameDiff.sameDiffFunctionDefinitionMap) : sameDiff.sameDiffFunctionDefinitionMap != null)
            return false;
        return sameDiffFunctionInstances != null ? sameDiffFunctionInstances.equals(sameDiff.sameDiffFunctionInstances) : sameDiff.sameDiffFunctionInstances == null;
    }

    
    public static SameDiff create() {
        return new SameDiff();
    }


    
    public INDArray[] eval(Map<String, INDArray> inputs) {

        SameDiff execPipeline = dup();

        List<DifferentialFunction> opExecAction = execPipeline.exec().getRight();
        if (opExecAction.isEmpty())
            throw new IllegalStateException("No ops found to execute.");
        INDArray[] ret = new INDArray[opExecAction.size()];
        for (int i = 0; i < ret.length; i++) {
            val varName = opExecAction.get(i).outputVariables()[0].getVarName();
            ret[i] = execPipeline.getArrForVarName(varName);
        }
        return ret;
    }




    
    public SameDiff dup() {
        Cloner cloner = newCloner();
        val clone = cloner.deepClone(this);


        return clone;

    }


    
    public long numElements() {
        long ret = 0;
        for (SDVariable variable : variables()) {
            ret += ArrayUtil.prod(variable.getShape());
        }

        return ret;
    }


    private void initWorkspace() {
        workspace = Nd4j.getWorkspaceManager().createNewWorkspace(
                WorkspaceConfiguration.builder()
                        .initialSize(memoryForGraph())
                        .policyAllocation(AllocationPolicy.OVERALLOCATE)
                        .policyLearning(LearningPolicy.FIRST_LOOP)
                        .build());
        Nd4j.getWorkspaceManager().setWorkspaceForCurrentThread(workspace);


    }


    
    public List<SDVariable> variables() {
        return new ArrayList<>(variableMap.values());
    }

    
    public SDVariable one(String name, int[] shape) {
        return var(name, ArrayUtil.toLongArray(shape), new ConstantInitScheme('f', 1.0));
    }

    public SDVariable one(String name, long[] shape) {
        return var(name, shape, new ConstantInitScheme('f', 1.0));
    }

    
    public SDVariable onesLike(SDVariable input) {
        return onesLike(null, input);
    }

    
    public SDVariable onesLike(String name, SDVariable input) {
        SDVariable ret = f().onesLike(name, input);
        return updateVariableNameAndReference(ret, name);
    }


    
    public SDVariable zero(String name, long[] shape) {
        return var(name, shape, new ZeroInitScheme());
    }

    public SDVariable zero(String name, int[] shape) {
        return var(name, ArrayUtil.toLongArray(shape), new ZeroInitScheme());
    }

    
    public SDVariable zerosLike(SDVariable input) {
        return zerosLike(null, input);
    }

    
    public SDVariable zerosLike(String name, SDVariable input) {
        SDVariable ret = f().zerosLike(name, input);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable constant(SDVariable value, long... shape) {
        return constant(null, value, shape);
    }

    public SDVariable constant(String name, SDVariable value, long... shape) {
        SDVariable ret = f().constant(value, shape);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable linspace(double start, double stop, long number) {
        return linspace(null, start, stop, number);
    }

    public SDVariable linspace(String name, double start, double stop, long number) {
        SDVariable ret = f().linspace(start, stop, number);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable[] meshgrid(SDVariable... inputs){
        return meshgrid(null, inputs);
    }

    public SDVariable[] meshgrid(List<String> names, SDVariable... inputs){
        return meshgrid(names, true, inputs);
    }

    public SDVariable[] meshgrid(List<String> names, boolean cartesian, SDVariable... inputs){
        Preconditions.checkState(names == null || names.size() == inputs.length,
                "Got %s names but %s inputs", (names == null ? 0 : names.size()), inputs.length);
        SDVariable[] ret = f().meshgrid(cartesian, inputs);
        for( int i=0; i<ret.length; i++ ){
            ret[i] = updateVariableNameAndReference(ret[i], names == null ? null : names.get(i));
        }
        return ret;
    }

    
    public SDVariable var(String name, long[] shape, WeightInitScheme weightInitScheme) {
        if (variableMap.containsKey(name) && variableMap.get(name).getArr() != null)
            throw new IllegalArgumentException("Another variable with the name " + name +
                    " already exists.");


        if (name == null || name.length() < 1)
            name = getNewVarName();

        if (workspace == null)
            initWorkspace();


        SDVariable ret = SDVariable.builder()
                .sameDiff(this)
                .shape(shape).weightInitScheme(weightInitScheme)
                .varName(name)
                .build();


        addVariable(ret);
        variableMap.put(name, ret);
        return ret;

    }


    
    public SDVariable var(String name, long... shape) {
        Preconditions.checkArgument(shape != null && shape.length > 0, "Invalid shape: %s", shape);
        return var(name, shape, new ZeroInitScheme());
    }

    public SDVariable var(String name, int... shape) {
        Preconditions.checkArgument(shape != null && shape.length > 0, "Invalid shape: %s", shape);
        return var(name, ArrayUtil.toLongArray(shape), new ZeroInitScheme());
    }


    
    public SDVariable var(final SDVariable arr) {
        if (variableMap.containsKey(arr.getVarName()) && variableMap.get(arr.getVarName()).getArr() != null)
            return variableMap.get(arr.getVarName());

        if (arr.getVarName() == null || arr.getVarName().length() < 1)
            throw new IllegalArgumentException("Name for variable must be defined");

        if (arr == null)
            throw new IllegalArgumentException("Array for " + arr.getVarName() + " must not be null");

        if (workspace == null)
            initWorkspace();

        final SDVariable ret = SDVariable.builder()
                .sameDiff(this)
                .shape(arr.getShape())
                .varName(arr.getVarName())
                .weightInitScheme(new NDArraySupplierInitScheme(new NDArraySupplierInitScheme.NDArraySupplier() {
                    @Override
                    public INDArray getArr() {
                        
                        if (arr.getArr() == null) {
                            INDArray retArr = arr.getWeightInitScheme().create(arr.getShape());
                            associateArrayWithVariable(retArr, arr);
                        }
                        return arr.getArr();
                    }
                }))
                .build();


        variableMap.put(arr.getVarName(), ret);
        return ret;

    }



    private int _var_id = 0;

    private String getNewVarName() {
        String varName = "sd_var_" + String.valueOf(_var_id);
        while (variableMap.containsKey(varName)) {
            _var_id++;
            varName = "sd_var_" + String.valueOf(_var_id);
        }
        return varName;
    }

    public SDVariable var(int... shape) {
        return var(getNewVarName(), shape);
    }

    public SDVariable var(long... shape) {
        return var(getNewVarName(), shape);
    }

    public SDVariable var(WeightInitScheme weightInitScheme, long... shape) {
        return var(getNewVarName(), shape, weightInitScheme);
    }

    public SDVariable var(INDArray arr) {
        return var(getNewVarName(), arr);
    }


    
    public SDVariable eye(int rows) {
        return eye(rows, rows);
    }

    
    public SDVariable eye(String name, int rows) {
        return eye(name, rows, rows);
    }

    
    public SDVariable eye(int rows, int cols) {
        return eye(null, rows, cols);
    }

    
    public SDVariable eye(String name, int rows, int cols) {
        return eye(name, rows, cols, null);
    }

    
    public SDVariable eye(int rows, int cols, int... batchDimension) {
        return eye(null, rows, cols, batchDimension);
    }

    
    public SDVariable eye(String name, int rows, int cols, int... batchDimension) {
        SDVariable eye = new Eye(this, rows, cols, batchDimension).outputVariables()[0];
        return updateVariableNameAndReference(eye, name);
    }


    
    public void removeArgFromFunction(String varName, DifferentialFunction function) {
        val args = function.args();

        for (int i = 0; i < args.length; i++) {
            if (args[i].getVarName().equals(varName)) {
                
                val reverseArgs = incomingArgsReverse.get(function.getOwnName());
                incomingArgsReverse.remove(function.getOwnName());
                val newArgs = new ArrayList<String>(args.length - 1);
                for (int arg = 0; arg < args.length; arg++) {
                    if (!reverseArgs[arg].equals(varName)) {
                        newArgs.add(reverseArgs[arg]);
                    }
                }

                val newArgsArr = newArgs.toArray(new String[newArgs.size()]);
                incomingArgsReverse.put(function.getOwnName(), newArgsArr);

                break;
            }
        }
    }


    
    public SDVariable var(String name, INDArray arr) {
        if (variableMap.containsKey(name) && variableMap.get(name).getArr() != null)
            throw new IllegalArgumentException("Another variable with the name " + name +
                    " already exists.");


        if (name == null || name.length() < 1)
            name = getNewVarName();

        if (arr == null)
            throw new IllegalArgumentException("Array for " + name + " must not be null");

        if (workspace == null)
            initWorkspace();

        val arrRef = arr.migrate();
        SDVariable ret = SDVariable.builder()
                .sameDiff(this)
                .shape(arr.shape())
                .varName(name)
                .weightInitScheme(new NDArraySupplierInitScheme(new NDArraySupplierInitScheme.NDArraySupplier() {
                    @Override
                    public INDArray getArr() {
                        return arrRef;
                    }
                }))
                .build();


        associateArrayWithVariable(arr, ret);
        if (ArrayUtil.prod(arr.shape()) == 1)
            ret.setScalarValue(arr.getDouble(0));

        addVariable(ret);
        if (getShapeForVarName(name) == null)
            putShapeForVarName(name, arr.shape());


        reverseArrayLookup.put(arr, ret);
        variableMap.put(name, ret);
        return ret;

    }

    
    public SDVariable getVariable(String name) {
        return variableMap.get(name);
    }


    
    public SDVariable getGradForVariable(String varName) {
        return gradients.get(varName);
    }


    
    public void setGradientForVariableName(String variableName, SDVariable variable) {
        if (variable == null) {
            throw new ND4JIllegalStateException("Unable to set null gradient for variable name " + variableName);
        }

        gradients.put(variableName, variable);
    }


    
    public SDVariable getForwardVariableForVertexId(int vertexId) {
        return forwardVarForGrad.get(vertexId);
    }


    
    public void setForwardVariableForVarName(String varName, SDVariable forwardVariable) {
        forwardVarForGrad.put(varName, forwardVariable);
    }

    
    public SDVariable grad(String varName) {
        if (!sameDiffFunctionInstances.containsKey("grad")) {
            throw new IllegalStateException("Unable to obtain gradient. Please run execBackwards() first.");
        }

        SameDiff grad = getFunction("grad");
        SDVariable var = grad.getVariable(varName);
        return getFunction("grad").getGradForVariable(var.getVarName());
    }

    public SDVariable randomUniform(double min, double max, SDVariable shape){
        return randomUniform(null, min, max, shape);
    }

    public SDVariable randomUniform(String name, double min, double max, SDVariable shape){
        SDVariable ret = f().randomUniform(min, max, shape);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable randomNormal(double mean, double stddev, SDVariable shape){
        return randomNormal(null, mean, stddev, shape);
    }

    public SDVariable randomNormal(String name, double mean, double stddev, SDVariable shape){
        SDVariable ret = f().randomNormal(mean, stddev, shape);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable randomBernoulli(double p, SDVariable shape){
        return randomBernoulli(null, p, shape);
    }

    public SDVariable randomBernoulli(String name, double p, SDVariable shape){
        SDVariable ret = f().randomBernoulli(p, shape);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable randomExponential(double lambda, SDVariable shape) {
        return randomExponential(null, lambda, shape);
    }

    
    public SDVariable randomExponential(String name, double lambda, SDVariable shape) {
        SDVariable ret = f().randomExponential(lambda, shape);
        return updateVariableNameAndReference(ret, name);
    }


    
    public SDVariable avgPooling2d(SDVariable input, Pooling2DConfig pooling2DConfig) {
        return avgPooling2d(null, input, pooling2DConfig);
    }

    
    public SDVariable avgPooling2d(String name, SDVariable input, Pooling2DConfig pooling2DConfig) {
        SDVariable ret = f().avgPooling2d(input, pooling2DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable maxPooling2d(SDVariable input, Pooling2DConfig pooling2DConfig) {
        return maxPooling2d(null, input, pooling2DConfig);
    }

    
    public SDVariable maxPooling2d(String name, SDVariable input, Pooling2DConfig pooling2DConfig) {
        SDVariable ret = f().maxPooling2d(input, pooling2DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable avgPooling3d(SDVariable input, Pooling3DConfig pooling3DConfig) {
        return avgPooling3d(null, input, pooling3DConfig);
    }

    
    public SDVariable avgPooling3d(String name, SDVariable input, Pooling3DConfig pooling3DConfig) {
        SDVariable ret = f().avgPooling3d(input, pooling3DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable maxPooling3d(SDVariable input, Pooling3DConfig pooling3DConfig) {
        return maxPooling3d(null, input, pooling3DConfig);
    }

    
    public SDVariable maxPooling3d(String name, SDVariable input, Pooling3DConfig pooling3DConfig) {
        SDVariable ret = f().maxPooling3d(input, pooling3DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable conv1d(SDVariable input, SDVariable weights, Conv1DConfig conv1DConfig) {
        return conv1d(null, input, weights, conv1DConfig);
    }

    
    public SDVariable conv1d(String name, SDVariable input, SDVariable weights, Conv1DConfig conv1DConfig) {
        SDVariable ret = f().conv1d(input, weights, conv1DConfig);
        return updateVariableNameAndReference(ret, name);
    }


    
    public SDVariable localResponseNormalization(SDVariable inputs, LocalResponseNormalizationConfig lrnConfig) {
        return localResponseNormalization(null, inputs, lrnConfig);
    }

    
    public SDVariable localResponseNormalization(String name, SDVariable input,
                                                 LocalResponseNormalizationConfig lrnConfig) {
        SDVariable ret = f().localResponseNormalization(input, lrnConfig);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable conv2d(SDVariable layerInput, SDVariable weights, Conv2DConfig config) {
        return conv2d(layerInput, weights, null, config);
    }


    
    public SDVariable conv2d(SDVariable layerInput, SDVariable weights, SDVariable bias, Conv2DConfig config) {
        SDVariable[] arr = new SDVariable[bias == null ? 2 : 3];
        arr[0] = layerInput;
        arr[1] = weights;
        if (bias != null)
            arr[2] = bias;
        return conv2d(arr, config);
    }

    
    public SDVariable conv2d(SDVariable[] inputs, Conv2DConfig conv2DConfig) {
        return conv2d(null, inputs, conv2DConfig);
    }

    
    public SDVariable conv2d(String name, SDVariable[] inputs, Conv2DConfig conv2DConfig) {
        SDVariable ret = f().conv2d(inputs, conv2DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable depthWiseConv2d(SDVariable layerInput, SDVariable depthWeights, Conv2DConfig config) {
        return depthWiseConv2d(layerInput, depthWeights, null, config);
    }


    
    public SDVariable depthWiseConv2d(SDVariable layerInput, SDVariable depthWeights, SDVariable bias, Conv2DConfig config) {
        SDVariable[] arr = new SDVariable[bias == null ? 2 : 3];
        arr[0] = layerInput;
        arr[1] = depthWeights;
        if (bias != null)
            arr[2] = bias;
        return depthWiseConv2d(arr, config);
    }


    
    public SDVariable depthWiseConv2d(SDVariable[] inputs, Conv2DConfig depthConv2DConfig) {
        return depthWiseConv2d(null, inputs, depthConv2DConfig);
    }


    
    public SDVariable depthWiseConv2d(String name, SDVariable[] inputs, Conv2DConfig depthConv2DConfig) {
        SDVariable ret = f().depthWiseConv2d(inputs, depthConv2DConfig);
        return updateVariableNameAndReference(ret, name);
    }


    
    public SDVariable separableConv2d(SDVariable layerInput, SDVariable depthWeights, SDVariable pointWeights,
                                      Conv2DConfig config) {
        return separableConv2d(layerInput, depthWeights, pointWeights, null, config);
    }


    
    public SDVariable separableConv2d(SDVariable layerInput, SDVariable depthWeights, SDVariable pointWeights,
                                      SDVariable bias, Conv2DConfig config) {
        SDVariable[] arr = new SDVariable[bias == null ? 3 : 4];
        arr[0] = layerInput;
        arr[1] = depthWeights;
        arr[2] = pointWeights;
        if (bias != null)
            arr[3] = bias;
        return sconv2d(arr, config);
    }

    
    public SDVariable sconv2d(SDVariable[] inputs, Conv2DConfig conv2DConfig) {
        return sconv2d(null, inputs, conv2DConfig);
    }


    
    public SDVariable sconv2d(String name, SDVariable[] inputs, Conv2DConfig conv2DConfig) {
        SDVariable ret = f().sconv2d(inputs, conv2DConfig);
        return updateVariableNameAndReference(ret, name);
    }


    
    public SDVariable deconv2d(SDVariable layerInput, SDVariable weights, DeConv2DConfig deconv2DConfig) {
        return deconv2d(layerInput, weights, null, deconv2DConfig);
    }


    
    public SDVariable deconv2d(SDVariable layerInput, SDVariable weights, SDVariable bias, DeConv2DConfig deconv2DConfig) {
        SDVariable[] arr = new SDVariable[bias == null ? 2 : 3];
        arr[0] = layerInput;
        arr[1] = weights;
        if (bias != null)
            arr[2] = bias;
        return deconv2d(arr, deconv2DConfig);
    }

    
    public SDVariable deconv2d(SDVariable[] inputs, DeConv2DConfig deconv2DConfig) {
        return deconv2d(null, inputs, deconv2DConfig);
    }


    
    public SDVariable deconv2d(String name, SDVariable[] inputs, DeConv2DConfig deconv2DConfig) {
        SDVariable ret = f().deconv2d(inputs, deconv2DConfig);
        return updateVariableNameAndReference(ret, name);
    }


    
    public SDVariable conv3d(SDVariable input, SDVariable weights, Conv3DConfig conv3DConfig) {
        return conv3d(null, input, weights, null, conv3DConfig);
    }

    
    public SDVariable conv3d(SDVariable input, SDVariable weights, SDVariable bias, Conv3DConfig conv3DConfig) {
        return conv3d(null, input, weights, bias, conv3DConfig);
    }

    
    public SDVariable conv3d(String name, SDVariable input, SDVariable weights, Conv3DConfig conv3DConfig) {
        return conv3d(null, input, weights, null, conv3DConfig);
    }

    
    public SDVariable conv3d(String name, SDVariable input, SDVariable weights, SDVariable bias, Conv3DConfig conv3DConfig) {
        SDVariable[] args;
        if (bias == null) {
            args = new SDVariable[]{input, weights};
        } else {
            args = new SDVariable[]{input, weights, bias};
        }
        SDVariable ret = f().conv3d(args, conv3DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable batchNorm(SDVariable input, SDVariable mean,
                                SDVariable variance, SDVariable gamma,
                                SDVariable beta,
                                boolean applyGamma, boolean applyBeta, double epsilon) {
        return batchNorm(null, input, mean, variance, gamma, beta, applyGamma, applyBeta, epsilon);
    }

    
    public SDVariable batchNorm(String name, SDVariable input, SDVariable mean,
                                SDVariable variance, SDVariable gamma,
                                SDVariable beta,
                                boolean applyGamma, boolean applyBeta, double epsilon) {
        SDVariable res = f().batchNorm(input, mean, variance, gamma, beta, applyGamma, applyBeta, epsilon);
        return updateVariableNameAndReference(res, name);
    }

    public SDVariable im2Col(SDVariable in, Conv2DConfig config) {
        return im2Col(null, in, config);
    }

    public SDVariable im2Col(String name, SDVariable in, Conv2DConfig config) {
        SDVariable ret = f().im2Col(in, config);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable col2Im(SDVariable in, Conv2DConfig config) {
        return col2Im(null, in, config);
    }

    public SDVariable col2Im(String name, SDVariable in, Conv2DConfig config) {
        SDVariable ret = f().col2Im(in, config);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable scalar(String name, double value) {
        return var(name, Nd4j.scalar(value));
    }


    
    public SDVariable gte(SDVariable iX, double iy) {
        return gte(null, iX, iy);

    }

    
    public SDVariable lte(SDVariable iX, double iy) {
        return lte(null, iX, iy);
    }


    
    public SDVariable gt(SDVariable iX, double iy) {
        return gt(null, iX, iy);
    }

    
    public SDVariable lt(SDVariable iX, double iy) {
        return lt(null, iX, iy);
    }


    
    public SDVariable neq(SDVariable iX, double iy) {
        return neq(null, iX, iy);
    }

    
    public SDVariable eq(SDVariable iX, double iy) {
        return eq(null, iX, iy);
    }

    
    public SDVariable gte(SDVariable iX, SDVariable iy) {
        return gte(null, iX, iy);
    }

    
    public SDVariable lte(SDVariable iX, SDVariable iy) {
        return lte(null, iX, iy);
    }


    
    public SDVariable gt(SDVariable iX, SDVariable iy) {
        return gt(null, iX, iy);

    }

    
    public SDVariable lt(SDVariable iX, SDVariable iy) {
        return lt(null, iX, iy);
    }


    
    public SDVariable neq(SDVariable iX, SDVariable iy) {
        return neq(null, iX, iy);
    }

    
    public SDVariable eq(SDVariable iX, SDVariable iy) {
        return eq(null, iX, iy);
    }

    
    public SDVariable or(SDVariable iX, SDVariable iy) {
        return or(null, iX, iy);
    }

    public SDVariable and(SDVariable iX, SDVariable iY) {
        return and(null, iX, iY);
    }

    public SDVariable and(String name, SDVariable ix, SDVariable iy) {
        SDVariable result = f().and(ix, iy);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable xor(SDVariable ix, SDVariable iy) {
        return xor(null, ix, iy);
    }

    public SDVariable xor(String name, SDVariable ix, SDVariable iy) {
        SDVariable result = f().xor(ix, iy);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable abs(SDVariable ix) {
        return abs(null, ix);
    }

    public SDVariable abs(String name, SDVariable ix) {
        SDVariable result = f().abs(ix);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable neg(SDVariable iX) {
        return neg(null, iX);
    }


    
    public SDVariable cos(SDVariable iX) {
        return cos(null, iX);
    }

    
    public SDVariable sin(SDVariable iX) {
        return sin(null, iX);
    }

    
    public SDVariable tan(SDVariable iX) {
        return tan(null, iX);
    }

    public SDVariable identity(SDVariable input) {
        return identity(null, input);
    }

    public SDVariable identity(String name, SDVariable input) {
        SDVariable s = f().identity(input);
        return updateVariableNameAndReference(s, name);
    }

    public SDVariable invertPermutation(SDVariable input) {
        return invertPermutation(null, input);
    }

    public SDVariable invertPermutation(String name, SDVariable input) {
        SDVariable ret = f().invertPermutation(input, false);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable acos(SDVariable iX) {
        return acos(null, iX);
    }

    

    public SDVariable asin(SDVariable iX) {
        return asin(null, iX);
    }

    
    public SDVariable atan(SDVariable iX) {
        return atan(null, iX);
    }

    public SDVariable atan2(SDVariable y, SDVariable x) {
        return atan2(null, y, x);
    }

    public SDVariable atan2(String name, SDVariable y, SDVariable x) {
        SDVariable ret = f().atan2(y, x);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable cosh(SDVariable iX) {
        return cosh(null, iX);
    }

    
    public SDVariable sinh(SDVariable iX) {
        return sinh(null, iX);
    }

    
    public SDVariable tanh(SDVariable iX) {
        return tanh(null, iX);
    }

    public SDVariable step(SDVariable in, double cutoff) {
        return step(null, in, cutoff);
    }

    public SDVariable step(String name, SDVariable in, double cutoff) {
        SDVariable ret = f().step(in, cutoff);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable acosh(SDVariable iX) {
        return acosh(null, iX);
    }

    
    public SDVariable asinh(SDVariable iX) {
        return asinh(null, iX);
    }

    
    public SDVariable atanh(SDVariable iX) {
        return atanh(null, iX);
    }

    
    public SDVariable exp(SDVariable iX) {
        return exp(null, iX);
    }


    
    public SDVariable rsqrt(SDVariable iX) {
        return rsqrt(null, iX);
    }

    
    public SDVariable expm1(SDVariable iX) {
        return expm1(null, iX);
    }

    
    public SDVariable log1p(SDVariable iX) {
        return log1p(null, iX);
    }


    
    public SDVariable isInfinite(SDVariable iX) {
        return isInfinite(null, iX);
    }

    
    public SDVariable isNaN(SDVariable iX) {
        return isNaN(null, iX);
    }

    
    public SDVariable round(SDVariable iX) {
        return round(null, iX);
    }

    
    public SDVariable isFinite(SDVariable iX) {
        return isFinite(null, iX);
    }

    public SDVariable isMax(SDVariable ix) {
        return isMax(null, ix);
    }

    public SDVariable isMax(String name, SDVariable ix) {
        SDVariable ret = f().isMax(ix);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable replaceWhere(SDVariable update, SDVariable from, Condition condition) {
        return replaceWhere(null, update, from, condition);
    }

    public SDVariable replaceWhere(String name, SDVariable update, SDVariable from, Condition condition) {
        SDVariable ret = f().replaceWhere(update, from, condition);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable replaceWhere(SDVariable update, Number value, Condition condition) {
        return replaceWhere(null, update, value, condition);
    }

    public SDVariable replaceWhere(String name, SDVariable to, Number value, Condition condition) {
        SDVariable ret = f().replaceWhere(to, value, condition);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable log(SDVariable iX) {
        return log(null, iX);
    }

    public SDVariable log(SDVariable in, double base) {
        return log(null, in, base);
    }

    public SDVariable log(String name, SDVariable in, double base) {
        SDVariable ret = f().log(in, base);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable logSumExp(SDVariable input, int... dimensions) {
        return logSumExp(null, input, dimensions);
    }

    public SDVariable logSumExp(String name, SDVariable input, int... dimensions) {
        SDVariable ret = f().logSumExp(input, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable cube(SDVariable iX) {
        return cube(null, iX);
    }


    
    public SDVariable pow(SDVariable iX, double value) {
        return pow(null, iX, value);
    }

    
    public SDVariable sqrt(SDVariable iX) {
        return sqrt(null, iX);
    }

    
    public SDVariable square(SDVariable iX) {
        return square(null, iX);
    }

    
    public SDVariable floor(SDVariable iX) {
        return floor(null, iX);
    }

    public SDVariable ceil(SDVariable x) {
        return ceil(null, x);
    }

    public SDVariable ceil(String name, SDVariable x) {
        SDVariable ret = f().ceil(x);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable clipByValue(SDVariable x, double clipValueMin, double clipValueMax) {
        return clipByValue(null, x, clipValueMin, clipValueMax);
    }

    public SDVariable clipByValue(String name, SDVariable x, double clipValueMin, double clipValueMax) {
        SDVariable ret = f().clipByValue(x, clipValueMin, clipValueMax);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable clipByNorm(SDVariable x, double clipValue) {
        return clipByNorm(null, x, clipValue);
    }

    public SDVariable clipByNorm(String name, SDVariable x, double clipValue) {
        SDVariable ret = f().clipByNorm(x, clipValue);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable clipByNorm(SDVariable x, double clipValue, int... dimensions) {
        return clipByNorm(null, x, clipValue, dimensions);
    }

    public SDVariable clipByNorm(String name, SDVariable x, double clipValue, int... dimensions) {
        SDVariable ret = f().clipByNorm(x, clipValue, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable relu(SDVariable iX, double cutoff) {
        return relu(null, iX, cutoff);
    }

    
    public SDVariable relu6(SDVariable iX, double cutoff) {
        return relu6(null, iX, cutoff);
    }

    
    public SDVariable softmax(SDVariable iX) {
        return softmax(null, iX);
    }

    public SDVariable logSoftmax(SDVariable iX) {
        return logSoftmax(null, iX);
    }

    public SDVariable logSoftmax(String name, SDVariable iX) {
        SDVariable ret = f().logSoftmax(iX);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable selu(SDVariable iX) {
        return selu(null, iX);
    }

    public SDVariable selu(String name, SDVariable iX) {
        SDVariable ret = f().selu(iX);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable mergeAdd(SDVariable... iX) {
        return mergeAdd(null, iX);
    }

    public SDVariable mergeAdd(String name, SDVariable... inputs) {
        SDVariable ret = f().mergeAdd(inputs);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable mergeMax(SDVariable... iX) {
        return mergeMax(null, iX);
    }

    public SDVariable mergeMax(String name, SDVariable... inputs) {
        SDVariable ret = f().mergeMax(inputs);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable mergeAvg(SDVariable... inputs) {
        return mergeAvg(null, inputs);
    }

    public SDVariable mergeAvg(String name, SDVariable... inputs) {
        SDVariable ret = f().mergeAvg(inputs);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable batchToSpace(SDVariable iX, int[] blocks, int[][] crops) {
        return batchToSpace(null, iX, blocks, crops);
    }

    public SDVariable batchToSpace(String name, SDVariable iX, int[] blocks, int[][] crops) {
        SDVariable ret = f().batchToSpace(iX, blocks, crops);
        return updateVariableNameAndReference(ret, name);
    }


    public SDVariable depthToSpace(SDVariable iX, int blockSize, String dataFormat) {
        return depthToSpace(null, iX, blockSize, dataFormat);
    }

    public SDVariable depthToSpace(String name, SDVariable iX, int blockSize, String dataFormat) {
        SDVariable ret = f().depthToSpace(iX, blockSize, dataFormat);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable spaceToBatch(SDVariable iX, int[] blocks, int[][] padding) {
        return spaceToBatch(null, iX, blocks, padding);
    }

    public SDVariable spaceToBatch(String name, SDVariable iX, int[] blocks, int[][] padding) {
        SDVariable ret = f().spaceToBatch(iX, blocks, padding);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable spaceToDepth(SDVariable iX, int blockSize, String dataFormat) {
        return spaceToDepth(null, iX, blockSize, dataFormat);
    }

    public SDVariable spaceToDepth(String name, SDVariable iX, int blockSize, String dataFormat) {
        SDVariable ret = f().spaceToDepth(iX, blockSize, dataFormat);
        return updateVariableNameAndReference(ret, name);
    }


    public SDVariable[] dynamicPartition(SDVariable iX, SDVariable partitions, int numPartitions) {
        return dynamicPartition(null, iX, partitions, numPartitions);
    }

    public SDVariable[] dynamicPartition(String[] name, SDVariable iX, SDVariable partitions, int numPartitions) {
        SDVariable[] ret = f().dynamicPartition(iX, partitions, numPartitions);
        return updateVariableNamesAndReferences(ret, name);
    }

    public SDVariable dynamicStitch(SDVariable[] indices, SDVariable[] iX) {
        return dynamicStitch(null, indices, iX);
    }

    public SDVariable dynamicStitch(String name, SDVariable[] indices, SDVariable[] iX) {
        SDVariable ret = f().dynamicStitch(indices, iX);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable dilation2D(SDVariable df, SDVariable weights, int[] strides,
                                 int[] rates, boolean isSameMode) {
        return dilation2D(null, df, weights, strides, rates, isSameMode);
    }

    public SDVariable dilation2D(String name, SDVariable df, SDVariable weights, int[] strides,
                                 int[] rates, boolean isSameMode) {
        SDVariable ret = f().dilation2D(df, weights, strides, rates, isSameMode);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable shape(SDVariable df) {
        return shape(null, df);
    }

    public SDVariable shape(String name, SDVariable df) {
        SDVariable ret = f().shape(df);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable size(SDVariable in){
        return size(null, in);
    }

    public SDVariable size(String name, SDVariable in){
        SDVariable ret = f().size(in);
        return updateVariableNameAndReference(ret, name);
    }
        return rank(null, in);
    }

    public SDVariable rank(String name, SDVariable in) {
        SDVariable ret = f().rank(in);
        return updateVariableNameAndReference(ret, name);
    }


    public SDVariable cross(SDVariable a, SDVariable b) {
        return cross(null, a, b);
    }

    public SDVariable cross(String name, SDVariable a, SDVariable b) {
        SDVariable ret = f().cross(a, b);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable gather(SDVariable df, int[] indices, int axis) {
        return gather(null, df, indices, axis);
    }

    public SDVariable gather(String name, SDVariable df, int[] indices, int axis) {
        SDVariable ret = f().gather(df, indices, axis);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable gather(SDVariable df, SDVariable indices, int axis) {
        return gather(null, df, indices, axis);
    }

    public SDVariable gather(String name, SDVariable df, SDVariable indices, int axis) {
        SDVariable ret = f().gather(df, indices, axis);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable gatherNd(SDVariable df, SDVariable indices) {
        return gatherNd(null, df, indices);
    }

    public SDVariable gatherNd(String name, SDVariable df, SDVariable indices) {
        SDVariable ret = f().gatherNd(df, indices);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable repeat(SDVariable df, int axis) {
        return repeat(null, df, axis);
    }


    public SDVariable repeat(String name, SDVariable df, int axis) {
        SDVariable ret = f().repeat(df, axis);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable stack(int axis, SDVariable... values) {
        return stack(null, axis, values);
    }

    public SDVariable stack(String name, int axis, SDVariable... values) {
        SDVariable ret = f().stack(values, axis);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable parallel_stack(SDVariable[] values) {
        return parallel_stack(null, values);
    }

    public SDVariable parallel_stack(String name, SDVariable[] values) {
        SDVariable ret = f().parallel_stack(values);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable[] unstack(SDVariable value, int axis) {
        return unstack(null, value, axis);
    }

    public SDVariable[] unstack(String[] names, SDVariable value, int axis) {
        SDVariable[] ret = f().unstack(value, axis);
        return updateVariableNamesAndReferences(ret, names);
    }

    public SDVariable[] unstack(SDVariable value, int axis, int num) {
        return unstack(null, value, axis, num);
    }

    public SDVariable[] unstack(String[] names, SDVariable value, int axis, int num) {
        SDVariable[] ret = f().unstack(value, axis, num);
        return updateVariableNamesAndReferences(ret, names);
    }

    public SDVariable erf(SDVariable iX) {
        return erf(null, iX);
    }

    public SDVariable erf(String name, SDVariable iX) {
        SDVariable ret = f().erf(iX);
        return updateVariableNameAndReference(ret, name);
    }


    public SDVariable erfc(SDVariable iX) {
        return erfc(null, iX);
    }

    public SDVariable erfc(String name, SDVariable iX) {
        SDVariable ret = f().erfc(iX);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable diag(SDVariable iX) {
        return diag(null, iX);
    }

    public SDVariable diag(String name, SDVariable iX) {
        SDVariable ret = f().diag(iX);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable diagPart(SDVariable iX) {
        return diagPart(null, iX);
    }

    public SDVariable diagPart(String name, SDVariable iX) {
        SDVariable ret = f().diagPart(iX);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable setDiag(SDVariable in, SDVariable diag) {
        return setDiag(null, in, diag);
    }

    public SDVariable setDiag(String name, SDVariable in, SDVariable diag) {
        SDVariable ret = f().setDiag(in, diag);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable oneHot(SDVariable indices, int depth) {
        return oneHot(null, indices, depth, -1, 1.00, 0.00);
    }

    public SDVariable oneHot(SDVariable indices, int depth, int axis, double on, double off) {
        return oneHot(null, indices, depth, axis, on, off);
    }

    public SDVariable oneHot(String name, SDVariable indices, int depth) {
        return oneHot(name, indices, depth, -1, 1.00, 0.00);
    }

    public SDVariable oneHot(String name, SDVariable indices, int depth, int axis, double on, double off) {
        SDVariable ret = f().onehot(indices, depth, axis, on, off);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable reciprocal(SDVariable a) {
        return reciprocal(null, a);
    }

    public SDVariable reciprocal(String name, SDVariable a) {
        SDVariable ret = f().reciprocal(a);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable gradientBackwardsMarker(SDVariable iX) {
        return gradientBackwardsMarker(generateNewVarName(new GradientBackwardsMarker().opName(), 0), iX);
    }


    
    public SDVariable hardTanh(SDVariable iX) {
        return hardTanh(null, iX);
    }

    public SDVariable hardSigmoid(SDVariable in) {
        return hardSigmoid(null, in);
    }

    public SDVariable hardSigmoid(String name, SDVariable in) {
        SDVariable ret = f().hardSigmoid(in);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable hardTanhDerivative(SDVariable iX) {
        return hardTanhDerivative(null, iX);
    }

    
    public SDVariable sigmoid(SDVariable iX) {
        return sigmoid(null, iX);
    }


    
    public SDVariable sigmoidDerivative(SDVariable iX, SDVariable wrt) {
        return sigmoidDerivative(null, iX, wrt);
    }

    public SDVariable logSigmoid(SDVariable iX) {
        return logSigmoid(null, iX);
    }

    public SDVariable logSigmoid(String name, SDVariable iX) {
        SDVariable ret = f().logSigmoid(iX);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable sign(SDVariable iX) {
        return sign(null, iX);
    }

    
    public SDVariable softsign(SDVariable iX) {
        return softsign(null, iX);
    }

    
    public SDVariable softsignDerivative(SDVariable iX) {
        return softsignDerivative(null, iX);
    }

    
    public SDVariable softplus(SDVariable iX) {
        return softplus(null, iX);
    }

    public SDVariable swish(SDVariable iX) {
        return swish(null, iX);
    }

    public SDVariable swish(String name, SDVariable iX) {
        SDVariable ret = f().swish(iX);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable elu(SDVariable iX) {
        return elu(null, iX);
    }

    
    public SDVariable eluDerivative(SDVariable iX) {
        return eluDerivative(null, iX);
    }

    
    public SDVariable leakyRelu(SDVariable iX, double cutoff) {
        return leakyRelu(null, iX, cutoff);
    }

    
    public SDVariable mean(SDVariable iX) {
        return mean(null, iX);
    }


    
    public SDVariable mean(SDVariable iX, int... dimension) {
        return mean(null, iX, dimension);
    }

    
    public SDVariable standardDeviation(SDVariable iX,
                                        boolean biasCorrected,
                                        int... dimensions) {
        return standardDeviation(null, iX, biasCorrected, dimensions);
    }

    
    public SDVariable variance(SDVariable iX,
                               boolean biasCorrected,
                               int... dimensions) {
        return variance(null, iX, biasCorrected, dimensions);
    }

    
    public SDVariable entropy(SDVariable in, int... dimensions) {
        return entropy(null, in, dimensions);
    }

    
    public SDVariable entropy(String name, SDVariable in, int... dimensions) {
        SDVariable ret = f().entropy(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable sum(SDVariable iX, int... dimensions) {
        return sum(null, iX, dimensions);
    }

    public SDVariable sum(SDVariable iX, boolean keepDims, int... dimensions) {
        return sum(null, iX, keepDims, dimensions);
    }



    
    public SDVariable prod(SDVariable iX, int... dimensions) {
        return prod(null, iX, dimensions);
    }


    public SDVariable scalarMax(SDVariable in, Number value) {
        return scalarMax(null, in, value);
    }

    public SDVariable scalarMax(String name, SDVariable in, Number value) {
        SDVariable ret = f().scalarMax(in, value);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable scalarMin(SDVariable in, Number value) {
        return scalarMin(null, in, value);
    }

    public SDVariable scalarMin(String name, SDVariable in, Number value) {
        SDVariable ret = f().scalarMin(in, value);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable scalarFloorMod(SDVariable in, Number value) {
        return scalarFloorMod(null, in, value);
    }

    public SDVariable scalarFloorMod(String name, SDVariable in, Number value) {
        SDVariable ret = f().scalarFloorMod(in, value);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable scalarSet(SDVariable in, Number set) {
        return scalarSet(null, in, set);
    }

    public SDVariable scalarSet(String name, SDVariable in, Number set) {
        SDVariable ret = f().scalarSet(in, set);
        return updateVariableNameAndReference(ret, name);
    }


    
    public SDVariable max(SDVariable iX, int... dimensions) {
        return max(null, iX, dimensions);
    }

    public SDVariable max(SDVariable first, SDVariable second) {
        return max(null, first, second);
    }

    public SDVariable max(String name, SDVariable first, SDVariable second) {
        SDVariable result = f().max(first, second);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable amax(SDVariable in, int... dimensions) {
        return amax(null, in, dimensions);
    }

    public SDVariable amax(String name, SDVariable in, int... dimensions) {
        SDVariable ret = f().amax(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable amin(SDVariable in, int... dimensions) {
        return amin(null, in, dimensions);
    }

    public SDVariable amin(String name, SDVariable in, int... dimensions) {
        SDVariable ret = f().amin(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable amean(SDVariable in, int... dimensions) {
        return amean(null, in, dimensions);
    }

    public SDVariable amean(String name, SDVariable in, int... dimensions) {
        SDVariable ret = f().amean(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable asum(SDVariable in, int... dimensions) {
        return asum(null, in, dimensions);
    }

    public SDVariable asum(String name, SDVariable in, int... dimensions) {
        SDVariable ret = f().asum(in, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable countZero(SDVariable input, int... dimensions) {
        return countZero(null, input, dimensions);
    }

    public SDVariable countZero(String name, SDVariable input, int... dimensions) {
        SDVariable res = f().countZero(input, dimensions);
        return updateVariableNameAndReference(res, name);
    }

    public SDVariable zeroFraction(SDVariable input) {
        return zeroFraction(null, input);
    }

    public SDVariable zeroFraction(String name, SDVariable input) {
        SDVariable res = f().zeroFraction(input);
        return updateVariableNameAndReference(res, name);
    }

    public SDVariable countNonZero(SDVariable input, int... dimensions) {
        return countNonZero(null, input, dimensions);
    }

    public SDVariable countNonZero(String name, SDVariable input, int... dimensions) {
        SDVariable res = f().countNonZero(input, dimensions);
        return updateVariableNameAndReference(res, name);
    }

    
    public SDVariable min(SDVariable iX, int... dimensions) {
        return min(null, iX, dimensions);
    }

    public SDVariable min(SDVariable first, SDVariable second) {
        return min(null, first, second);
    }

    public SDVariable min(String name, SDVariable first, SDVariable second) {
        SDVariable result = f().min(first, second);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable argmax(SDVariable in, int... dimensions) {
        return argmax(null, in, false, dimensions);
    }

    public SDVariable argmax(SDVariable in, boolean keepDims, int... dimensions) {
        return argmax(null, in, dimensions);
    }

    public SDVariable argmax(String name, SDVariable in, int... dimensions) {
        return argmax(name, in, false, dimensions);
    }

    public SDVariable argmax(String name, SDVariable in, boolean keepDims, int... dimensions) {
        SDVariable ret = f().argmax(in, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable argmin(SDVariable in, int... dimensions) {
        return argmin(null, in, dimensions);
    }

    public SDVariable argmin(SDVariable in, boolean keepDims, int... dimensions) {
        return argmin(null, in, keepDims, dimensions);
    }

    public SDVariable argmin(String name, SDVariable in, int... dimensions) {
        return argmin(name, in, false, dimensions);
    }

    public SDVariable argmin(String name, SDVariable in, boolean keepDims, int... dimensions) {
        SDVariable ret = f().argmin(in, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable iamax(SDVariable in, int... dimensions) {
        return iamax(null, in, dimensions);
    }

    public SDVariable iamax(SDVariable in, boolean keepDims, int... dimensions) {
        return iamax(null, in, keepDims, dimensions);
    }

    public SDVariable iamax(String name, SDVariable in, int... dimensions) {
        return iamax(name, in, false, dimensions);
    }

    public SDVariable iamax(String name, SDVariable in, boolean keepDims, int... dimensions) {
        SDVariable ret = f().iamax(in, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable iamin(SDVariable in, int... dimensions) {
        return iamin(null, in, dimensions);
    }

    public SDVariable iamin(SDVariable in, boolean keepDims, int... dimensions) {
        return iamin(null, in, keepDims, dimensions);
    }

    public SDVariable iamin(String name, SDVariable in, int... dimensions) {
        return iamin(name, in, false, dimensions);
    }

    public SDVariable iamin(String name, SDVariable in, boolean keepDims, int... dimensions) {
        SDVariable ret = f().iamin(in, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable firstIndex(SDVariable in, Condition condition, int... dimensions) {
        return firstIndex(null, in, condition, dimensions);
    }

    public SDVariable firstIndex(SDVariable in, Condition condition, boolean keepDims, int... dimensions){
        return firstIndex(null, in, condition, keepDims, dimensions);
    }

    public SDVariable firstIndex(String name, SDVariable in, Condition condition, int... dimensions) {
        return firstIndex(name, in, condition, false, dimensions);
    }

    public SDVariable firstIndex(String name, SDVariable in, Condition condition, boolean keepDims, int... dimensions){
        SDVariable ret = f().firstIndex(in, condition, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable lastIndex(SDVariable in, Condition condition, int... dimensions) {
        return lastIndex(null, in, condition, dimensions);
    }

    public SDVariable lastIndex(SDVariable in, Condition condition, boolean keepDims, int... dimensions){
        return lastIndex(null, in, condition, keepDims, dimensions);
    }

    public SDVariable lastIndex(String name, SDVariable in, Condition condition, int... dimensions) {
        return lastIndex(name, in, condition, false, dimensions);
    }

    public SDVariable lastIndex(String name, SDVariable in, Condition condition, boolean keepDims, int... dimensions){
        SDVariable ret = f().lastIndex(in, condition, keepDims, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable matchCondition(SDVariable in, Condition condition) {
        return matchCondition(null, in, condition);
    }

    public SDVariable matchCondition(String name, SDVariable in, Condition condition) {
        SDVariable ret = f().matchCondition(in, condition);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable cumsum(SDVariable in, SDVariable axis, boolean exclusive, boolean reverse) {
        return cumsum(null, in, axis, exclusive, reverse);
    }

    public SDVariable cumsum(String name, SDVariable in, SDVariable axis, boolean exclusive, boolean reverse) {
        SDVariable ret = f().cumsum(in, axis, exclusive, reverse);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable cumprod(SDVariable in, SDVariable axis, boolean exclusive, boolean reverse) {
        return cumprod(null, in, axis, exclusive, reverse);
    }

    public SDVariable cumprod(String name, SDVariable in, SDVariable axis, boolean exclusive, boolean reverse) {
        SDVariable ret = f().cumprod(in, axis, exclusive, reverse);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable biasAdd(SDVariable input, SDVariable bias) {
        return biasAdd(null, input, bias);
    }

    public SDVariable biasAdd(String name, SDVariable input, SDVariable bias) {
        SDVariable ret = f().biasAdd(input, bias);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable reshape(SDVariable iX, int... shape) {
        return reshape(null, iX, shape);
    }

    public SDVariable reshape(SDVariable iX, SDVariable shape) {
        return reshape(null, iX, shape);
    }


    
    public SDVariable reverse(SDVariable x, int... dimensions) {
        return reverse(null, x, dimensions);
    }

    
    public SDVariable reverse(String name, SDVariable x, int... dimensions) {
        SDVariable ret = f().reverse(x, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable reverseSequence(String name, SDVariable x, SDVariable seq_lengths, int seqDim, int batchDim) {
        SDVariable ret = f().reverseSequence(x, seq_lengths, seqDim, batchDim);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable reverseSequence(String name, SDVariable x, SDVariable seq_lengths) {
        SDVariable ret = f().reverseSequence(x, seq_lengths);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable reverseSequence(SDVariable x, SDVariable seq_lengths, int seqDim, int batchDim) {
        return reverseSequence(null, x, seq_lengths, seqDim, batchDim);
    }

    public SDVariable reverseSequence(SDVariable x, SDVariable seq_lengths) {
        return reverseSequence(null, x, seq_lengths);
    }

    public SDVariable sequenceMask(String name, SDVariable lengths, SDVariable maxLen) {
        SDVariable ret = f().sequenceMask(lengths, maxLen);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable sequenceMask(SDVariable lengths, SDVariable maxLen) {
        return sequenceMask(null, lengths, maxLen);
    }

    public SDVariable sequenceMask(String name, SDVariable lengths, int maxLen) {
        SDVariable ret = f().sequenceMask(lengths, maxLen);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable sequenceMask(SDVariable lengths, int maxLen) {
        return sequenceMask(null, lengths, maxLen);
    }

    public SDVariable sequenceMask(String name, SDVariable lengths) {
        SDVariable ret = f().sequenceMask(lengths);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable sequenceMask(SDVariable lengths) {
        SDVariable ret = f().sequenceMask(lengths);
        return updateVariableNameAndReference(ret, null);
    }

    public SDVariable assign(SDVariable x, SDVariable y) {
        return assign(null, x, y);
    }

    public SDVariable assign(String name, SDVariable x, SDVariable y) {
        SDVariable ret = f().assign(x, y);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable assign(SDVariable in, Number value) {
        return assign(null, in, value);
    }

    public SDVariable assign(String name, SDVariable in, Number value) {
        SDVariable ret = f().assign(in, value);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable transpose(SDVariable iX) {
        return transpose(null, iX);
    }

    
    public SDVariable permute(SDVariable iX, int... dimensions) {
        return permute(null, iX, dimensions);
    }

    
    public SDVariable rollAxis(SDVariable x, int axis) {
        return rollAxis(null, x, axis);
    }

    
    public SDVariable concat(int dimension, SDVariable... inputs) {
        return concat(null, dimension, inputs);
    }

    public SDVariable[] moments(SDVariable input, int... axes) {
        return moments(null, input, axes);
    }

    public SDVariable[] moments(String[] name, SDVariable input, int... axes) {
        SDVariable[] res = f().moments(input, axes);
        return updateVariableNamesAndReferences(res, name);
    }

    public SDVariable[] normalizeMoments(SDVariable counts, SDVariable means, SDVariable variances, double shift) {
        return normalizeMoments(null, counts, means, variances, shift);
    }

    public SDVariable[] normalizeMoments(String[] name, SDVariable counts, SDVariable means, SDVariable variances,
                                         double shift) {
        SDVariable[] res = f().normalizeMoments(counts, means, variances, shift);
        return updateVariableNamesAndReferences(res, name);
    }

    
    public SDVariable tile(SDVariable iX, int[] repeat) {
        return tile(null, iX, repeat);
    }

    public SDVariable fill(SDVariable shape, double value) {
        return fill(null, shape, value);
    }

    public SDVariable dropout(SDVariable input, double p) {
        return dropout(null, input, p);
    }

    public SDVariable dropout(String name, SDVariable input, double p) {
        SDVariable res = f().dropout(input, p);
        return updateVariableNameAndReference(res, name);
    }


    public SDVariable xwPlusB(SDVariable input, SDVariable weights, SDVariable bias) {
        return xwPlusB(null, input, weights, bias);
    }

    public SDVariable xwPlusB(String name, SDVariable input, SDVariable weights, SDVariable bias) {
        SDVariable res = f().xwPlusB(input, weights, bias);
        return updateVariableNameAndReference(res, name);
    }


    public SDVariable reluLayer(SDVariable input, SDVariable weights, SDVariable bias) {
        return reluLayer(null, input, weights, bias);
    }

    public SDVariable reluLayer(String name, SDVariable input, SDVariable weights, SDVariable bias) {
        SDVariable res = f().reluLayer(input, weights, bias);
        return updateVariableNameAndReference(res, name);
    }

    
    public SDVariable mmul(SDVariable x, SDVariable y, MMulTranspose transpose) {
        return mmul(null, x, y, transpose);

    }

    
    public SDVariable mmul(SDVariable x, SDVariable y) {
        return mmul(null, x, y);
    }

    
    public SDVariable tensorMmul(SDVariable x,
                                 SDVariable y,
                                 int[][] dimensions) {
        return tensorMmul(null, x, y, dimensions);
    }


    public SDVariable dot(SDVariable x, SDVariable y, int... dimensions) {
        return dot(null, x, y, dimensions);
    }

    public SDVariable dot(String name, SDVariable x, SDVariable y, int... dimensions) {
        SDVariable ret = f().dot(x, y, dimensions);
        return updateVariableNameAndReference(ret, name);
    }

    
    public SDVariable cosineSimilarity(SDVariable iX, SDVariable i_y, int... dimensions) {
        return cosineSimilarity(generateNewVarName(CosineSimilarity.OP_NAME, 0), iX, i_y, dimensions);
    }

    
    public SDVariable euclideanDistance(SDVariable iX, SDVariable i_y, int... dimensions) {
        return euclideanDistance(generateNewVarName(EuclideanDistance.OP_NAME, 0), iX, i_y, dimensions);
    }

    
    public SDVariable manhattanDistance(SDVariable iX, SDVariable i_y, int... dimensions) {
        return manhattanDistance(generateNewVarName(ManhattanDistance.OP_NAME, 0), iX, i_y, dimensions);
    }

    public SDVariable cosineDistance(SDVariable ix, SDVariable iy, int... dimensions) {
        return cosineDistance(null, ix, iy, dimensions);
    }

    public SDVariable cosineDistance(String name, SDVariable ix, SDVariable iy, int... dimensions) {
        SDVariable result = functionFactory.cosineDistance(ix, iy, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable hammingDistance(SDVariable ix, SDVariable iy, int... dimensions) {
        return hammingDistance(null, ix, iy, dimensions);
    }

    public SDVariable hammingDistance(String name, SDVariable ix, SDVariable iy, int... dimensions) {
        SDVariable result = functionFactory.hammingDistance(ix, iy, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable jaccardDistance(SDVariable ix, SDVariable iy, int... dimensions) {
        return jaccardDistance(null, ix, iy, dimensions);
    }

    public SDVariable jaccardDistance(String name, SDVariable ix, SDVariable iy, int... dimensions) {
        SDVariable result = functionFactory.jaccardDistance(ix, iy, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable lossBinaryXENT(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossBinaryXENT(generateNewVarName(new LossBinaryXENT().opName(), 0), iX, i_y, dimensions);
    }

    
    public SDVariable lossCosineSimilarity(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossCosineSimilarity(generateNewVarName(new LossCosineProximity().opName(), 0), iX, i_y, dimensions);
    }

    
    public SDVariable lossHinge(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossHinge(generateNewVarName(new LossHinge().opName(), 0), iX, i_y, dimensions);

    }

    
    public SDVariable lossKLD(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossKLD(generateNewVarName(new LossKLD().opName(), 0), iX, i_y, dimensions);

    }

    
    public SDVariable lossL1(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossL1(generateNewVarName(new LossL1().opName(), 0), iX, i_y, dimensions);

    }

    
    public SDVariable lossL2(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossL2(generateNewVarName(new LossL2().opName(), 0), iX, i_y, dimensions);

    }

    
    public SDVariable lossMAE(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossMAE(generateNewVarName(new LossMAE().opName(), 0), iX, i_y, dimensions);

    }

    
    public SDVariable lossMSE(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossMSE(generateNewVarName(new LossMSE().opName(), 0), iX, i_y, dimensions);

    }

    
    public SDVariable lossMCXENT(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossMCXENT(generateNewVarName(new LossMCXENT().opName(), 0), iX, i_y, dimensions);

    }

    
    public SDVariable lossMSLE(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossMSLE(generateNewVarName(new LossMSLE().opName(), 0), iX, i_y, dimensions);

    }

    
    public SDVariable lossNegativeLogLikelihood(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossNegativeLogLikelihood(generateNewVarName(new LossNegativeLogLikelihood().opName(), 0), iX, i_y, dimensions);

    }

    
    public SDVariable lossPoisson(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossPoisson(generateNewVarName(new LossPoisson().opName(), 0), iX, i_y, dimensions);

    }


    
    public SDVariable lossSquaredHinge(SDVariable iX, SDVariable i_y, int... dimensions) {
        return lossSquaredHinge(generateNewVarName(new LossSquaredHinge().opName(), 0), iX, i_y, dimensions);
    }


    
    public SDVariable gradientBackwardsMarker(String name, SDVariable iX) {
        SDVariable result = functionFactory.gradientBackwardsMarker(iX);
        return updateVariableNameAndReference(result, name);
    }


    
    public SDVariable neq(String name, SDVariable iX, double iy) {
        SDVariable result = functionFactory.neq(iX, iy);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable eq(String name, SDVariable iX, double iy) {
        SDVariable result = functionFactory.eq(iX, iy);
        return updateVariableNameAndReference(result, name);

    }


    
    public SDVariable gte(String name, SDVariable iX, double iy) {
        SDVariable result = functionFactory.gte(iX, iy);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable lte(String name, SDVariable iX, double iy) {
        SDVariable result = functionFactory.lte(iX, iy);
        return updateVariableNameAndReference(result, name);

    }


    
    public SDVariable gt(String name, SDVariable iX, double iy) {
        SDVariable result = functionFactory.gt(iX, iy);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable lt(String name, SDVariable iX, double iy) {
        SDVariable result = functionFactory.lt(iX, iy);
        return updateVariableNameAndReference(result, name);

    }


    
    public SDVariable neq(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.neq(iX, iy);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable eq(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.eq(iX, iy);
        return updateVariableNameAndReference(result, name);

    }


    
    public SDVariable gte(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.gte(iX, iy);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable lte(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.lte(iX, iy);
        return updateVariableNameAndReference(result, name);

    }


    
    public SDVariable gt(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.gt(iX, iy);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable lt(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.lt(iX, iy);
        return updateVariableNameAndReference(result, name);

    }


    
    public SDVariable or(String name, SDVariable iX, SDVariable iy) {
        SDVariable result = functionFactory.or(iX, iy);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable neg(String name, SDVariable iX) {
        SDVariable result = functionFactory.neg(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable isNonDecreasing(SDVariable iX) {
        return isNonDecreasing(null, iX);

    }

    
    public SDVariable isNonDecreasing(String name, SDVariable iX) {
        SDVariable result = functionFactory.isNonDecreasing(iX);
        return updateVariableNameAndReference(result, name);

    }


    
    public SDVariable isStrictlyIncreasing(SDVariable iX) {
        return isStrictlyIncreasing(null, iX);

    }

    
    public SDVariable isStrictlyIncreasing(String name, SDVariable iX) {
        SDVariable result = functionFactory.isStrictlyIncreasing(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable isNumericTensor(SDVariable iX) {
        return isNumericTensor(null, iX);

    }

    
    public SDVariable isNumericTensor(String name, SDVariable iX) {
        SDVariable result = functionFactory.isNumericTensor(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable cos(String name, SDVariable iX) {
        SDVariable result = functionFactory.cos(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable sin(String name, SDVariable iX) {
        SDVariable result = functionFactory.sin(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable tan(String name, SDVariable iX) {
        SDVariable result = functionFactory.tan(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable acos(String name, SDVariable iX) {
        SDVariable result = functionFactory.acos(iX);
        return updateVariableNameAndReference(result, name);

    }

    

    public SDVariable asin(String name, SDVariable iX) {
        SDVariable result = functionFactory.asin(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable atan(String name, SDVariable iX) {
        SDVariable result = functionFactory.atan(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable cosh(String name, SDVariable iX) {
        SDVariable result = functionFactory.cosh(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable sinh(String name, SDVariable iX) {
        SDVariable result = functionFactory.sinh(iX);
        return updateVariableNameAndReference(result, name);


    }

    
    public SDVariable tanh(String name, SDVariable iX) {
        SDVariable
                result = functionFactory.tanh(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable acosh(String name, SDVariable iX) {
        SDVariable result = functionFactory.acosh(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable asinh(String name, SDVariable iX) {
        SDVariable result = functionFactory.asinh(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable atanh(String name, SDVariable iX) {
        SDVariable result = functionFactory.atanh(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable exp(String name, SDVariable iX) {
        SDVariable result = functionFactory.exp(iX);
        return updateVariableNameAndReference(result, name);

    }


    
    public SDVariable expm1(String name, SDVariable iX) {
        SDVariable result = functionFactory.expm1(iX);
        return updateVariableNameAndReference(result, name);
    }


    
    public SDVariable rsqrt(String name, SDVariable iX) {
        SDVariable result = functionFactory.rsqrt(iX);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable log(String name, SDVariable iX) {
        SDVariable result = functionFactory.log(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable log1p(String name, SDVariable iX) {
        SDVariable result = functionFactory.log1p(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable isFinite(String name, SDVariable iX) {
        SDVariable result = functionFactory.isFinite(iX);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable isInfinite(String name, SDVariable iX) {
        SDVariable result = functionFactory.isInfinite(iX);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable isNaN(String name, SDVariable iX) {
        SDVariable result = functionFactory.isNaN(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable round(String name, SDVariable iX) {
        SDVariable result = functionFactory.round(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable pow(String name, SDVariable iX, double value) {
        SDVariable result = functionFactory.pow(iX, value);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable cube(String name, SDVariable iX) {
        SDVariable result = functionFactory.cube(iX);
        return updateVariableNameAndReference(result, name);

    }


    
    public SDVariable sqrt(String name, SDVariable iX) {
        SDVariable result = functionFactory.sqrt(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable square(String name, SDVariable iX) {
        SDVariable result = functionFactory.square(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable floor(String name, SDVariable iX) {
        SDVariable result = functionFactory.floor(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable relu(String name, SDVariable iX, double cutoff) {
        SDVariable result = functionFactory.relu(iX, cutoff);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable relu6(String name, SDVariable iX, double cutoff) {
        SDVariable result = functionFactory.relu6(iX, cutoff);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable softmax(String name, SDVariable iX) {
        SDVariable result = functionFactory.softmax(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable softmaxDerivative(String name, SDVariable iX, SDVariable wrt) {
        SDVariable result = functionFactory.softmaxDerivative(iX, wrt);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable hardTanh(String name, SDVariable iX) {
        SDVariable result = functionFactory.hardTanh(iX);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable hardTanhDerivative(String name, SDVariable iX) {
        SDVariable result = functionFactory.hardTanhDerivative(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable sigmoid(String name, SDVariable iX) {
        SDVariable result = functionFactory.sigmoid(iX);
        return updateVariableNameAndReference(result, name);

    }


    
    public SDVariable sigmoidDerivative(String name, SDVariable iX, SDVariable wrt) {
        SDVariable result = functionFactory
                .sigmoidDerivative(iX, wrt);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable sign(String name, SDVariable iX) {
        SDVariable result = functionFactory
                .sign(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable softsign(String name, SDVariable iX) {
        SDVariable result = functionFactory.softsign(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable softsignDerivative(String name, SDVariable iX) {
        SDVariable result = functionFactory.softsignDerivative(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable softplus(String name, SDVariable iX) {
        SDVariable result = functionFactory.softplus(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable elu(String name, SDVariable iX) {
        SDVariable result = functionFactory.elu(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable eluDerivative(String name, SDVariable iX) {
        SDVariable result = functionFactory.eluDerivative(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable leakyRelu(String name, SDVariable iX, double alpha) {
        SDVariable result = functionFactory.leakyRelu(iX, alpha);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable leakyReluDerivative(String name, SDVariable iX, double alpha) {
        SDVariable result = functionFactory.leakyReluDerivative(iX, alpha);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable mean(String name, SDVariable iX, int... dimension) {
        return mean(name, iX, false, dimension);
    }

    public SDVariable mean(String name, SDVariable iX, boolean keepDims, int... dimension) {
        SDVariable result = functionFactory.mean(iX, keepDims, dimension);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable standardDeviation(String name, SDVariable iX,
                                        boolean biasCorrected,
                                        int... dimensions) {
        return standardDeviation(name, iX, biasCorrected, false, dimensions);
    }

    public SDVariable standardDeviation(String name, SDVariable iX,
                                        boolean biasCorrected,
                                        boolean keepDims,
                                        int... dimensions) {
        SDVariable result = functionFactory.std(
                iX,
                biasCorrected,
                keepDims,
                dimensions);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable variance(String name, SDVariable iX, boolean biasCorrected, int... dimensions) {
        return variance(name, iX, biasCorrected, false, dimensions);
    }

    public SDVariable variance(String name, SDVariable iX, boolean biasCorrected, boolean keepDims, int... dimensions) {
        SDVariable result = functionFactory.variance(iX, biasCorrected, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable sum(String name, SDVariable iX, int... dimensions) {
        return sum(name, iX, false, dimensions);
    }

    public SDVariable sum(String name, SDVariable iX, boolean keepDims, int... dimensions) {
        SDVariable result = functionFactory.sum(iX, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable prod(String name, SDVariable iX, int... dimensions) {
        return prod(name, iX, false, dimensions);
    }

    public SDVariable prod(String name, SDVariable iX, boolean keepDims, int... dimensions) {
        SDVariable result = functionFactory.prod(iX, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);

    }


    
    public SDVariable max(String name, SDVariable iX, int... dimensions) {
        return max(name, iX, false, dimensions);
    }

    public SDVariable max(String name, SDVariable iX, boolean keepDims, int... dimensions) {
        SDVariable result = functionFactory.max(iX, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);

    }


    
    public SDVariable min(String name, SDVariable iX, int... dimensions) {
        return min(name, iX, false, dimensions);
    }

    public SDVariable min(String name, SDVariable iX, boolean keepDims, int... dimensions) {
        SDVariable result = functionFactory.min(iX, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);

    }

    public SDVariable norm1(String name, SDVariable ix, int... dimensions) {
        return norm1(name, ix, false, dimensions);
    }

    public SDVariable norm1(String name, SDVariable ix, boolean keepDims, int... dimensions) {
        SDVariable result = f().norm1(ix, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable norm2(String name, SDVariable ix, int... dimensions) {
        return norm2(name, ix, false, dimensions);
    }

    public SDVariable norm2(String name, SDVariable ix, boolean keepDims, int... dimensions) {
        SDVariable result = f().norm2(ix, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable squaredNorm(SDVariable ix, int... dimensions) {
        return squaredNorm(null, ix, false, dimensions);
    }

    public SDVariable squaredNorm(String name, SDVariable ix, int... dimensions) {
        return squaredNorm(name, ix, false, dimensions);
    }

    public SDVariable squaredNorm(SDVariable ix, boolean keepDims, int... dimensions) {
        return squaredNorm(null, ix, keepDims, dimensions);
    }

    public SDVariable squaredNorm(String name, SDVariable ix, boolean keepDims, int... dimensions) {
        SDVariable result = f().squaredNorm(ix, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable normmax(String name, SDVariable ix, int... dimensions) {
        return normmax(name, ix, false, dimensions);
    }

    public SDVariable normmax(String name, SDVariable ix, boolean keepDims, int... dimensions) {
        SDVariable result = f().normmax(ix, keepDims, dimensions);
        return updateVariableNameAndReference(result, name);
    }


    
    public SDVariable reshape(String name, SDVariable iX,
                              int... shape) {
        SDVariable result = functionFactory
                .reshape(iX, shape);
        return updateVariableNameAndReference(result, name);

    }

    public SDVariable reshape(String name, SDVariable iX,
                              SDVariable shape) {
        SDVariable result = functionFactory
                .reshape(iX, shape);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable transpose(String name, SDVariable iX) {
        SDVariable result = functionFactory.transpose(iX);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable permute(String name, SDVariable iX, int... dimensions) {
        SDVariable result = functionFactory.permute(iX, dimensions);
        return updateVariableNameAndReference(result, name);

    }


    
    public SDVariable rollAxis(String name, SDVariable x, int axis) {
        SDVariable result = functionFactory.rollAxis(x, axis);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable fill(String name, SDVariable shape, double value) {
        SDVariable result = functionFactory.fill(shape, value);
        return updateVariableNameAndReference(result, name);

    }


    
    public SDVariable concat(String name, int dimension, SDVariable... inputs) {
        SDVariable result = functionFactory.concat(dimension, inputs);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable tile(String name, SDVariable iX, int[] repeat) {
        SDVariable result = functionFactory.tile(iX, repeat);
        return updateVariableNameAndReference(result, name);

    }


    
    public SDVariable mmul(String name, SDVariable x, SDVariable y, MMulTranspose transpose) {
        SDVariable result = functionFactory.mmul(x, y, transpose);
        return updateVariableNameAndReference(result, name);

    }

    
    public SDVariable mmul(String name, SDVariable x, SDVariable y) {
        return mmul(name, x, y, MMulTranspose.allFalse());
    }

    
    public SDVariable tensorMmul(String name,
                                 SDVariable x,
                                 SDVariable y,
                                 int[][] dimensions) {
        SDVariable result = functionFactory.tensorMmul(x, y, dimensions);
        return updateVariableNameAndReference(result, name);
    }


    
    public SDVariable cosineSimilarity(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable cosim = functionFactory.cosineSimilarity(
                iX,
                i_y,
                dimensions);
        return updateVariableNameAndReference(cosim, name);
    }

    
    public SDVariable euclideanDistance(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.euclideanDistance(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable manhattanDistance(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.manhattanDistance(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable sigmoidCrossEntropyWithLogits(SDVariable logits, SDVariable weights, SDVariable labels,
                                                    int reductionMode, double labelSmoothing) {
        return sigmoidCrossEntropyWithLogits(null, logits, weights, labels, reductionMode, labelSmoothing);
    }

    public SDVariable sigmoidCrossEntropyWithLogits(String name, SDVariable logits, SDVariable weights, SDVariable labels,
                                                    int reductionMode, double labelSmoothing) {
        SDVariable res = f().sigmoidCrossEntropyWithLogits(logits, weights, labels, reductionMode, labelSmoothing);
        return updateVariableNameAndReference(res, name);
    }

    public SDVariable softmaxCrossEntropyWithLogits(SDVariable logits, SDVariable weights, SDVariable labels,
                                                    int reductionMode, double labelSmoothing) {
        return softmaxCrossEntropyWithLogits(null, logits, weights, labels, reductionMode, labelSmoothing);
    }

    public SDVariable softmaxCrossEntropyWithLogits(String name, SDVariable logits, SDVariable weights, SDVariable labels,
                                                    int reductionMode, double labelSmoothing) {
        SDVariable res = f().softmaxCrossEntropyWithLogits(logits, weights, labels, reductionMode, labelSmoothing);
        return updateVariableNameAndReference(res, name);
    }

    public SDVariable weightedCrossEntropyWithLogits(SDVariable targets, SDVariable inputs,
                                                     SDVariable weights) {
        return weightedCrossEntropyWithLogits(null, targets, inputs, weights);
    }

    public SDVariable weightedCrossEntropyWithLogits(String name, SDVariable targets, SDVariable inputs,
                                                     SDVariable weights) {
        SDVariable res = f().weightedCrossEntropyWithLogits(targets, inputs, weights);
        return updateVariableNameAndReference(res, name);
    }

    
    public SDVariable lossBinaryXENT(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossBinaryXENT(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable lossCosineSimilarity(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossCosineSimilarity(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable lossHinge(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossHinge(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable lossKLD(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossKLD(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable lossL1(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossL1(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable lossL2(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossL2(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable lossMAE(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossMAE(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }


    
    public SDVariable lossMSE(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossMSE(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable lossMCXENT(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossMCXENT(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable lossMSLE(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossMSLE(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable lossNegativeLogLikelihood(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossNegativeLogLikelihood(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }

    
    public SDVariable lossPoisson(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossPoisson(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }


    
    public SDVariable lossSquaredHinge(String name, SDVariable iX, SDVariable i_y, int... dimensions) {
        SDVariable result = functionFactory.lossSquaredHinge(iX, i_y, dimensions);
        return updateVariableNameAndReference(result, name);
    }


    public SDVariable expandDims(SDVariable ix, int axis) {
        return expandDims(null, ix, axis);
    }

    public SDVariable expandDims(String name, SDVariable ix, int axis) {
        SDVariable result = f().expandDims(ix, axis);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable squeeze(SDVariable ix, int axis) {
        return squeeze(null, ix, axis);
    }

    public SDVariable squeeze(String name, SDVariable ix, int axis) {
        SDVariable result = f().squeeze(ix, axis);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable confusionMatrix(SDVariable labels, SDVariable predictions) {
        return confusionMatrix((String) null, labels, predictions);
    }

    public SDVariable confusionMatrix(String name, SDVariable labels, SDVariable pred) {
        SDVariable result = f().confusionMatrix(labels, pred);
        return updateVariableNameAndReference(result, name);
    }


    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, Integer numClasses) {
        return confusionMatrix(null, labels, pred, numClasses);
    }

    public SDVariable confusionMatrix(String name, SDVariable labels, SDVariable pred, Integer numClasses) {
        SDVariable result = f().confusionMatrix(labels, pred, numClasses);
        return updateVariableNameAndReference(result, name);
    }

    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, SDVariable weights) {
        return confusionMatrix(null, labels, pred, weights);
    }

    public SDVariable confusionMatrix(String name, SDVariable labels, SDVariable pred, SDVariable weights) {
        SDVariable result = f().confusionMatrix(labels, pred, weights);
        return updateVariableNameAndReference(result, name);
    }


    public SDVariable confusionMatrix(SDVariable labels, SDVariable pred, Integer numClasses, SDVariable weights) {
        return confusionMatrix(null, labels, pred, numClasses, weights);
    }

    public SDVariable confusionMatrix(String name, SDVariable labels, SDVariable pred, Integer numClasses, SDVariable weights) {
        SDVariable result = f().confusionMatrix(labels, pred, numClasses, weights);
        return updateVariableNameAndReference(result, name);
    }

    
    public void addVariable(SDVariable variable) {
        if (variableMap == null)
            variableMap = new HashMap<>();

        Preconditions.checkState(variable.getSameDiff() == this, "Samediff instance must be the same.");


        
        if (variableMap.containsKey(variable.getVarName()) && !variableMap.get(variable.getVarName()).equals(variable)) {
            throw new IllegalArgumentException("Variable already found with variable opName " + variable.getVarName());
        }

        Preconditions.checkState(variable.getSameDiff() == this, "Same diff instance for variable must be the same!");
        variableMap.put(variable.getVarName(), variable);

    }


    
    public String generateNewVarName(String baseName, int argIndex) {
        if (getVariable(baseName) == null && argIndex == 0) {
            return baseName;
        }


        int count = 1;
        String name = baseName + "_" + count + (argIndex > 0 ? ":" + argIndex : "");
        while (getVariable(name) != null) {
            name = baseName + "_" + (++count) + (argIndex > 0 ? ":" + argIndex : "");
        }

        if (getVariable(name) != null) {
            throw new ND4JIllegalStateException("Converged on already generated variable!");
        }

        return name;
    }


    
    public SDVariable lstm(String baseName, LSTMCellConfiguration configuration) {
        return new LSTMCell(this, configuration).outputVariables(baseName)[0];
    }


    
    public SDVariable sruCell(SRUCellConfiguration configuration) {
        return new SRUCell(this, configuration).outputVariables()[0];
    }


    
    public SDVariable sru(SRUConfiguration configuration) {
        return new SRU(this, configuration).outputVariables()[0];
    }

    
    public SDVariable gru(GRUCellConfiguration configuration) {
        return new GRUCell(this, configuration).outputVariables()[0];
    }


    
    public SDVariable sruCell(String baseName, SRUCellConfiguration configuration) {
        return new SRUCell(this, configuration).outputVariables(baseName)[0];
    }


    
    public SDVariable sru(String baseName, SRUConfiguration configuration) {
        return new SRU(this, configuration).outputVariables(baseName)[0];
    }

    
    public SDVariable gru(String baseName, GRUCellConfiguration configuration) {
        return new GRUCell(this, configuration).outputVariables(baseName)[0];
    }


    public SDVariable slice(SDVariable input, int[] begin, int[] size) {
        return slice(null, input, begin, size);
    }

    public SDVariable slice(String name, SDVariable input, int[] begin, int[] size) {
        SDVariable ret = f().slice(input, begin, size);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable stridedSlice(SDVariable input, int[] begin, int[] end, int[] strides) {
        return stridedSlice(null, input, begin, end, strides);
    }

    public SDVariable stridedSlice(String name, SDVariable input, int[] begin, int[] end, int[] strides) {
        return stridedSlice(name, input, begin, end, strides, 0, 0, 0, 0, 0);
    }

    public SDVariable stridedSlice(SDVariable input, long[] begin, long[] end, long[] strides) {
        return stridedSlice(null, input, begin, end, strides);
    }

    public SDVariable stridedSlice(String name, SDVariable input, long[] begin, long[] end, long[] strides) {
        return stridedSlice(name, input, begin, end, strides, 0, 0, 0, 0, 0);
    }

    public SDVariable stridedSlice(SDVariable in, int[] begin, int[] end, int[] strides, int beginMask,
                                   int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        return stridedSlice(null, in, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
    }

    public SDVariable stridedSlice(String name, SDVariable in, int[] begin, int[] end, int[] strides, int beginMask,
                                   int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        SDVariable ret = f().stridedSlice(in, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable stridedSlice(SDVariable in, long[] begin, long[] end, long[] strides, int beginMask,
                                   int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        return stridedSlice(null, in, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
    }

    public SDVariable stridedSlice(String name, SDVariable in, long[] begin, long[] end, long[] strides, int beginMask,
                                   int endMask, int ellipsisMask, int newAxisMask, int shrinkAxisMask) {
        SDVariable ret = f().stridedSlice(in, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable scatterAdd(String name, SDVariable ref, SDVariable indices, SDVariable updates) {
        SDVariable ret = f().scatterAdd(ref, indices, updates);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable scatterMul(String name, SDVariable ref, SDVariable indices, SDVariable updates) {
        SDVariable ret = f().scatterMul(ref, indices, updates);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable scatterSub(String name, SDVariable ref, SDVariable indices, SDVariable updates) {
        SDVariable ret = f().scatterSub(ref, indices, updates);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable scatterDiv(String name, SDVariable ref, SDVariable indices, SDVariable updates) {
        SDVariable ret = f().scatterDiv(ref, indices, updates);
        return updateVariableNameAndReference(ret, name);
    }

    public SDVariable scatterUpdate(SDVariable ref, SDVariable indices, SDVariable updates) {
        return scatterUpdate(null, ref, indices, updates);
    }

    public SDVariable scatterUpdate(String name, SDVariable ref, SDVariable indices, SDVariable updates) {
        SDVariable ret = f().scatterUpdate(ref, indices, updates);
        return updateVariableNameAndReference(ret, name);
    }


    public SDVariable scatterAdd(SDVariable ref, SDVariable indices, SDVariable updates) {
        return scatterAdd(null, ref, indices, updates);
    }

    public SDVariable scatterMul(SDVariable ref, SDVariable indices, SDVariable updates) {
        return scatterMul(null, ref, indices, updates);
    }

    public SDVariable scatterSub(SDVariable ref, SDVariable indices, SDVariable updates) {
        return scatterSub(null, ref, indices, updates);
    }

    public SDVariable scatterDiv(SDVariable ref, SDVariable indices, SDVariable updates) {
        return scatterDiv(null, ref, indices, updates);
    }


    
    public SDVariable[] generateOutputVariableForOp(DifferentialFunction function, String baseName) {


        if (baseName == null || baseName.isEmpty() && getBaseNameForFunction(function) != null)
            baseName = getBaseNameForFunction(function);

        if (baseName == null)
            baseName = function.opName();

        val outputShape = function.calculateOutputShape();
        if (outputShape == null || outputShape.isEmpty()) {
            if (function instanceof CustomOp) {
                CustomOp customOp = (CustomOp) function;

                int num_outputs = function.getNumOutputs(); 
                if (num_outputs <= 0) {
                    val descriptor = customOp.getDescriptor();
                    if (descriptor != null) {
                        num_outputs = descriptor.getNumOutputs();
                    }
                    if (num_outputs <= 0) {
                        throw new ND4UnresolvedOutputVariables("Could not determine number of output variables for op "
                                + function.getOwnName() + " - " + function.getClass().getSimpleName() + ". Ops can override" +
                                " getNumOutputs() to specify number of outputs if required");
                    }
                }
                char ordering = 'c';
                SDVariable[] args = function.args();
                if (args != null && args.length > 0 && args[0].getArr() != null) {  
                    ordering = function.args()[0].getArr().ordering();
                }
                SDVariable[] ret = new SDVariable[num_outputs];

                for (int i = 0; i < ret.length; i++) {
                    SDVariable checkGet = getVariable(baseName);
                    if (checkGet == null) {
                        checkGet = var(generateNewVarName(baseName, i), null, new ZeroInitScheme(ordering));
                    } else if (i > 0 && !importedVarName.contains(baseName)) {

                        String newName = generateNewVarName(baseName, i);
                        checkGet = getVariable(newName);
                    }
                    if (checkGet == null) {
                        String newName = generateNewVarName(baseName, i);
                        checkGet = var(newName, null, new ZeroInitScheme(ordering));
                    }
                    checkGet.setOutputIndex(i);
                    checkGet.setCreator(function);
                    ret[i] = checkGet;
                }


                if (getOutputsForFunction(function) == null)
                    addOutgoingFor(ret, function);

                return ret;
            }


            else if (function instanceof BaseOp && outputShape.isEmpty()) {
                SDVariable[] ret = new SDVariable[1];
                SDVariable checkGet = getVariable(baseName);
                char ordering = 'c';
                SDVariable[] args = function.args();
                if (args != null && args.length > 0 && function.args()[0].getArr() != null) { 
                    ordering = function.args()[0].getArr().ordering();
                }
                if (checkGet == null) {
                    checkGet = var(baseName, null, new ZeroInitScheme(ordering));
                } else if (!importedVarName.contains(baseName)) {

                    String newName = generateNewVarName(baseName, 0);
                    checkGet = var(newName, null, new ZeroInitScheme(ordering));
                }


                if (checkGet == null) {
                    checkGet = var(baseName, null, new ZeroInitScheme(ordering));
                }

                checkGet.setOutputIndex(0);
                checkGet.setCreator(function);
                ret[0] = checkGet;



                if (getOutputsForFunction(function) == null)
                    addOutgoingFor(ret, function);

                return ret;

            }
        }


        char ordering = 'c';
        if (function.args() != null && function.args().length > 0 && function.args()[0].getArr() != null) {
            ordering = function.args()[0].getArr().ordering();
        }

        SDVariable[] ret = new SDVariable[outputShape.size()];


        val ownName = function.getOwnName();
        val rootName = baseName;
        for (int i = 0; i < ret.length; i++) {
            val shape = outputShape.get(i);

            baseName = rootName + (i > 0 ? ":" + i : "");
            SDVariable checkGet = getVariable(baseName);
            if (checkGet == null) {

                checkGet = var(baseName, shape, new ZeroInitScheme(ordering));
            } else if (shape != null && !shapeAlreadyExistsForVarName(checkGet.getVarName())) {

                putShapeForVarName(checkGet.getVarName(), shape);
            } else if (shape != null && shapeAlreadyExistsForVarName(checkGet.getVarName())) {

                

            } else if (!importedVarName.contains(baseName)) {


                int count = 1;
                String name = baseName + "_" + count + (i > 0 ? ":" + i : "");
                while (getVariable(name) != null) {
                    count++;
                    name = baseName + "_" + count + (i > 0 ? ":" + i : "");
                }

                if (getVariable(name) != null) {
                    throw new ND4JIllegalStateException("Converged on already generated variable!");
                }


                checkGet = var(name, shape, new ZeroInitScheme(ordering));
            }

            if (checkGet == null) {
                checkGet = var(baseName + (i > 0 ? ":" + i : ""), shape, new ZeroInitScheme(ordering));
            }

            checkGet.setOutputIndex(i);
            checkGet.setCreator(function);
            ret[i] = checkGet;
        }


        return ret;
    }

    
    public SDVariable[] generateOutputVariableForOp(DifferentialFunction function) {
        return generateOutputVariableForOp(function, function.opName());
    }


    
    public SameDiff getFunction(String functionName) {
        return sameDiffFunctionInstances.get(functionName);
    }


    
    public INDArray execAndEndResult(List<DifferentialFunction> ops) {
        List<DifferentialFunction> exec = exec(ops);
        Op op = (Op) exec.get(exec.size() - 1);
        return op.z();
    }

    
    public INDArray execAndEndResult() {
        List<DifferentialFunction> exec = exec().getRight();
        val finalOp = exec.get(exec.size() - 1);
        val output = finalOp.outputVariables();
        if (output.length > 1) {
            throw new ND4JIllegalStateException(finalOp.opName() + " has multiple outputs. Use execAndEndResults instead.");
        }
        return output[0].getArr();
    }

    public INDArray[] execAndEndResults() {
        List<DifferentialFunction> exec = exec().getRight();
        val finalOp = exec.get(exec.size() - 1);
        val output = finalOp.outputVariables();
        INDArray outArrays[] = new INDArray[output.length];
        for (int i = 0; i < outArrays.length; i++) {
            outArrays[i] = output[i].getArr();
        }
        return outArrays;
    }

    public INDArray execAndEndResult(int outputIndex) {
        List<DifferentialFunction> exec = exec().getRight();
        val output = exec.get(exec.size() - 1).outputVariables()[outputIndex];
        return output.getArr();
    }


    public INDArray yetAnotherExecMethod(@NonNull Map<String, INDArray> inputs) {
        if (!wasRegistered.get()) {
            synchronized (this) {
                if (!wasRegistered.get()) {
                    val bb = asFlatBuffers();
                    val ptr = new BytePointer(bb);

                    Nd4j.getExecutioner().registerGraph(this.hashCode(), ptr);

                    wasRegistered.set(true);
                }
            }
        }

        val newMap = new LinkedHashMap<String, INDArray>();
        val keySet = inputs.keySet();

        for (val key : keySet) {
            val vx = variableMap.get(key);
            newMap.put(vx.getVarName(), inputs.get(key));
        }

        val result = Nd4j.getExecutioner().executeGraph(this.hashCode(), newMap, this.reverseMap);
        if (result.size() == 0)
            throw new ND4JIllegalStateException("Execution failed");

        val list = new ArrayList<INDArray>(result.values());

        return list.get(list.size() - 1);
    }


    
    public List<DifferentialFunction> exec(List<DifferentialFunction> ops) {
        for (int i = 0; i < ops.size(); i++) {
            Op op = (Op) ops.get(i);
            Nd4j.getExecutioner().exec(op);
        }
        return ops;
    }

    public TensorList getListByName(@NonNull String name) {
        return lists.get(name);
    }

    public void putListByName(@NonNull String name, TensorList list) {
        lists.put(name, list);
    }

    
    public interface SameDiffConditional {


        
        SDVariable eval(SameDiff context, SameDiffFunctionDefinition body, SDVariable[] inputVars);

    }

    public static class DefaultSameDiffConditional implements SameDiffConditional {

        @Override
        public SDVariable eval(SameDiff context, SameDiff.SameDiffFunctionDefinition body, SDVariable[] inputVars) {
            context.defineFunction("eval", body, inputVars);
            context.invokeFunctionOn("eval", context);
            return new ArrayList<>(context.functionInstancesById.values()).get(context.functionInstancesById.size() - 1).outputVariables()[0];
        }
    }


    
    public While whileStatement(SameDiffConditional sameDiffConditional,
                                SameDiffFunctionDefinition conditionBody,
                                SameDiff.SameDiffFunctionDefinition loopBody
            , SDVariable[] inputVars) {
        return While.builder()
                .inputVars(inputVars)
                .condition(conditionBody)
                .predicate(sameDiffConditional)
                .trueBody(loopBody)
                .parent(this)
                .blockName("while-" + UUID.randomUUID().toString())
                .build();
    }

    
    public If ifStatement(SameDiffConditional conditional,
                          SameDiffFunctionDefinition conditionBody,
                          SameDiffFunctionDefinition trueBody,
                          SameDiffFunctionDefinition falseBody
            , SDVariable[] inputVars) {
        return If.builder()
                .conditionBody(conditionBody)
                .falseBody(falseBody)
                .trueBody(trueBody)
                .predicate(conditional)
                .inputVars(inputVars)
                .parent(this)
                .blockName("if-" + UUID.randomUUID().toString())
                .build();
    }


    public TensorArrayV3 tensorArray() {
        return new TensorArrayV3(this);
    }

    
    public interface SameDiffFunctionDefinition {

        
        SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs);
    }

    

    public SDVariable invokeFunctionOn(String functionName, SameDiff with) {
        SameDiff instance = sameDiffFunctionInstances.get(functionName);
        SDVariable ret = instance.invokeGraphOn(with);

        return ret;
    }


    
    public SameDiff defineFunction(String function, SameDiffFunctionDefinition functionDefinition, SDVariable[] variables) {
        if (!sameDiffFunctionInstances.containsKey(function)) {
            SameDiff sub = SameDiff.create();
            sub.workspace = (workspace);
            this.child = sub;
            sub.parent = this;


            SDVariable[] ret = new SDVariable[variables.length];
            for (int i = 0; i < ret.length; i++) {
                ret[i] = sub.var(variables[i]);
            }

            sub.inputs = ret;
            sub.outputs = functionDefinition.define(sub, null, ret);

            sameDiffFunctionInstances.put(function, sub);
        }
        this.child = null;
        return sameDiffFunctionInstances.get(function);
    }


    
    public void defineFunction(String function, SameDiffFunctionDefinition functionDefinition) {
        defineFunction(function, functionDefinition, new LinkedHashMap<String, INDArray>());
    }

    
    public void defineFunction(String function,
                               SameDiffFunctionDefinition functionDefinition,
                               Map<String, INDArray> inputs) {
        if (!sameDiffFunctionInstances.containsKey(function)) {
            SameDiff sub = SameDiff.create();
            sub.workspace = (workspace);


            functionDefinition.define(sub, inputs, null);

            sameDiffFunctionInstances.put(function, sub);
        }

    }


    
    public INDArray execAndEndResult(String functionName) {
        return sameDiffFunctionInstances.get(functionName).execAndEndResult();
    }


    
    public Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec(String functionName) {
        if (debugMode) {
            return sameDiffFunctionInstances.get(functionName).enableDebugMode().exec();
        } else
            return sameDiffFunctionInstances.get(functionName).exec();
    }

    
    public List<DifferentialFunction> exec(String functionName, List<DifferentialFunction> cachedOps) {
        return sameDiffFunctionInstances.get(functionName).exec(cachedOps);
    }


    
    public Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> execBackwards() {
        if (getFunction("grad") == null) {
            createGradFunction();
        }


        if (log.isTraceEnabled()) {
            log.trace("About to execute backward function");
        }
        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> forward = exec("grad");
        SameDiff grad = getFunction("grad");
        if (grad.isDebugMode()) {

            for (SDVariable sdVariable : grad.variables()) {
                sdVariable.gradient();
            }
        }

        return forward;
    }

    public void createGradFunction() {
        if (log.isTraceEnabled()) {
            log.trace("Defining function \"grad\"");
        }

        final SameDiff outer = this;
        defineFunction("grad", new SameDiffFunctionDefinition() {

            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {


                if (SameDiff.this.debugMode) {
                    sameDiff.enableDebugMode();
                }

                outer.invokeGraphOn(sameDiff);
                if (debugMode) {

                    Preconditions.checkState(sameDiff.incomingArgsReverse.keySet().equals(incomingArgsReverse.keySet()), "incomingArgsReverse keysets not equal");
                    Preconditions.checkState(sameDiff.outgoingArgsReverse.keySet().equals(outgoingArgsReverse.keySet()), "outgoingArgsReverse keysets not equal");
                }

                List<DifferentialFunction> allFunctions = new ArrayList<>(sameDiff.functionInstancesById.values());
                if (allFunctions.isEmpty()) {
                    throw new ND4JIllegalStateException("No ops found!");
                }


                for (val func : allFunctions) {
                    if (func instanceof SDVariable) {
                        continue;
                    }

                    val args = func.args();
                    for (val arg : args)
                        arg.setSameDiff(sameDiff);
                    val outputs = func.outputVariables();
                    for (val output : outputs)
                        output.setSameDiff(sameDiff);
                    func.setSameDiff(sameDiff);
                }

                val initialOuts = allFunctions.get(allFunctions.size() - 1).outputVariables();
                val firstBackward = initialOuts[0];

                if (log.isTraceEnabled()) {
                    String[] initialOutputsStr = allFunctions.get(allFunctions.size() - 1).outputVariablesNames();
                    String s = initialOutputsStr == null ? "null" : Arrays.toString(initialOutputsStr);
                    log.trace("Defining backward function: initial outputs {}", s);
                }


                SDVariable initialGrad = sameDiff.var("one-var", Nd4j.trueScalar(1.0));
                sameDiff.forwardVarForGrad.put(firstBackward.getVarName(), initialGrad);
                sameDiff.gradients.put(firstBackward.getVarName(), initialGrad);

                SDVariable gradientBackwardsMarker = sameDiff.gradientBackwardsMarker(firstBackward);


                allFunctions = new ArrayList<>(sameDiff.functionInstancesById.values());
                Collections.reverse(allFunctions);


                for (int i = 0; i < allFunctions.size(); i++) {
                    DifferentialFunction action = allFunctions.get(i);
                    if (log.isTraceEnabled()) {
                        log.trace("Defining backward function step {} of {}: {} ({}) - {}", (i + 1), allFunctions.size(),
                                action.opName(), action.getOwnName(), action.getClass().getName());
                    }

                    if (action instanceof GradientBackwardsMarker) {
                        continue;
                    }

                    DifferentialFunction currFunction = action;
                    Preconditions.checkState(currFunction.getSameDiff() == sameDiff, "Wrong samediff instance found!");

                    val args = currFunction.outputVariables();
                    for (val arg : args) {
                        if (arg.getSameDiff() != sameDiff) {
                            arg.setSameDiff(sameDiff);
                        }
                    }


                    List<SDVariable> grads = new ArrayList<>();
                    for (val varToGrad : args) {
                        val grad = varToGrad.gradient();
                        if (grad == null)
                            throw new ND4JIllegalStateException("No gradient found for " + varToGrad.getVarName());
                        grads.add(grad);
                    }

                    List<SDVariable> currFnGrads = currFunction.diff(grads);

                    if (log.isTraceEnabled()) {
                        log.trace("Finished Defining backward function step {} of {}: {} ({}) - {}", (i + 1), allFunctions.size(),
                                action.opName(), action.getOwnName(), action.getClass().getName());
                    }

                    if (debugMode) {

                        Preconditions.checkState(sameDiff.incomingArgsReverse.keySet().equals(sameDiff.outgoingArgsReverse.keySet()),
                                "incomingArgsReverse and outgoingArgsReverse keysets not equal after backprop of function %s of %s: %s (%s)",
                                (i + 1), allFunctions.size(), action.getOwnName(), action.getClass().getName());
                    }
                }


                if (sameDiff.isDebugMode()) {

                    for (SDVariable sdVariable : variables()) {
                        sdVariable.gradient();
                    }
                }

                if (log.isTraceEnabled()) {
                    log.trace("Defining backward function complete");
                }

                return new SDVariable[]{sameDiff.var("grad", new int[]{1, 1})};
            }
        });
    }


    
    public INDArray execBackwardAndEndResult() {
        List<DifferentialFunction> backwards = execBackwards().getRight();
        DifferentialFunction df = backwards.get(backwards.size() - 1);
        if (df instanceof Op) {
            return ((Op) df).z();
        } else if (df instanceof DynamicCustomOp) {
            return ((DynamicCustomOp) df).getOutputArgument(0);
        } else {
            return null;
        }
    }


    
    public INDArray execWithPlaceHolderAndEndResult(Map<String, INDArray> inputs) {
        resolveVariablesWith(inputs);
        return execAndEndResult();
    }


    
    public void setOriginalPlaceHolderShape(String variableName, long[] shape) {
        if (!isPlaceHolder(variableName)) {
            throw new ND4JIllegalStateException("Vertex id " + variableName + " does not appear to be a place holder. Did you forget to call addPlaceHolder?");
        }

        if (shape == null) {
            throw new ND4JIllegalStateException("Null and 0 length shape arrays not allowed");
        }


        if (placeHolderOriginalShapes.containsKey(variableName) && !Arrays.equals(placeHolderOriginalShapes.get(variableName), shape)) {
            throw new ND4JIllegalStateException("Unable to add a new shape for vertex id " + variableName);
        }


        placeHolderOriginalShapes.put(variableName, shape);

    }


    
    public long[] getOriginalShapeForPlaceHolder(String varName) {
        return placeHolderOriginalShapes.get(varName);
    }

    
    public boolean isPlaceHolder(String varName) {
        return placeHolderVarNames.contains(varName);
    }


    
    public void addAsPlaceHolder(String varName) {
        placeHolderVarNames.add(varName);
        if (getVariable(varName) != null && getVariable(varName).getShape() != null) {
            placeHolderOriginalShapes.put(varName, getVariable(varName).getShape());
        }
    }


    
    public void resolveVariablesWith(Map<String, INDArray> arrays) {
        for (val arrayEntry : arrays.entrySet()) {
            val varForName = getVariable(arrayEntry.getKey());
            if (varForName == null) {
                throw new ND4JIllegalStateException("No variable name found for " + arrayEntry.getKey());
            }

            if (placeHolderOriginalShapes.containsKey(arrayEntry.getKey())) {
                val originalShape = placeHolderOriginalShapes.get(arrayEntry.getKey());
                if (originalShape.length == arrayEntry.getValue().rank()) {
                    for (int i = 0; i < originalShape.length; i++) {
                        if (originalShape[i] != arrayEntry.getValue().shape()[i] && originalShape[i] >= 1) {
                            throw new ND4JIllegalStateException("Incompatible shape passed for variable. " + Arrays.toString(arrayEntry.getValue().shape()));
                        }
                    }
                }
            }
        }


        for (val entry : arrays.entrySet()) {
            if (!placeHolderVarNames.contains(entry.getKey())) {
                throw new ND4JIllegalStateException("Illegal variable " + entry.getKey() + " passed in. Variable found not to be a place holder variable");
            }

            val specifiedShape = getOriginalShapeForPlaceHolder(entry.getKey());

            if (!Shape.isPlaceholderShape(specifiedShape)) {
                if (!Shape.shapeEquals(specifiedShape, entry.getValue().shape())) {
                    throw new ND4JIllegalStateException("Place holder shape specified was " + Arrays.toString(specifiedShape) + " but array shape was " + Arrays.toString(entry.getValue().shape()));
                }
            }


            updateShapeForVarName(entry.getKey(), entry.getValue().shape());
            associateArrayWithVariable(entry.getValue(), getVariable(entry.getKey()));
            updateArrayForVarName(entry.getKey(), entry.getValue());

        }


        for (val funcName : propertiesToResolve.keySet()) {
            val func = functionInstancesById.get(funcName);
            if (!functionInstancesById.containsKey(funcName)) {
                throw new ND4JIllegalStateException("Unable to resolve function name " + funcName);
            }

            if (func instanceof CustomOp) {
                CustomOp customOp = (CustomOp) func;
                customOp.populateInputsAndOutputsFromSameDiff();
            }

        }



        resolvedVariables = true;
    }

    
    public boolean allPlaceHolderVariablesResolved() {
        for (val vertexId : placeHolderVarNames) {
            val var = getVariable(vertexId);
            if (var.getArr() == null) {
                return false;
            }
        }

        return true;
    }

    
    public void putPlaceHolderForVariable(String varName, String... placeHolderVariables) {
        for (val placeHolderVariable : placeHolderVariables) {
            if (!variableMap.containsKey(placeHolderVariable)) {
                throw new ND4JIllegalStateException("No variable found for " + placeHolderVariable);
            }
        }


        List<String[]> placeHolders = placeHolderMap.get(varName);
        if (placeHolders == null) {
            placeHolders = new ArrayList<>();
            placeHolderMap.put(varName, placeHolders);
        }

        placeHolders.add(placeHolderVariables);
    }


    
    public boolean hasPlaceHolderVariables(String vertexId) {
        return placeHolderMap.containsKey(vertexId);
    }

    
    public List<String[]> getPlaceHoldersFor(String varName) {
        return placeHolderMap.get(varName);
    }


    
    public Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> execWithPlaceHolder(Map<String, INDArray> inputs) {
        resolveVariablesWith(inputs);
        return exec();
    }

    
    public List<SDVariable> getVariablesAssociatedWithFunctions(List<DifferentialFunction> functions) {
        List<SDVariable> ret = new ArrayList<>(functions.size());
        for (DifferentialFunction function : functions) {
            ret.addAll(Arrays.asList(function.outputVariables()));
        }

        return ret;
    }


    
    public SDVariable updateVariableNameAndReference(SDVariable varToUpdate, String newVarName) {
        if (varToUpdate == null) {
            throw new NullPointerException("Null input: No variable found for updating!");
        }

        if (newVarName == null && variableMap.containsKey(varToUpdate.getVarName())) {


            newVarName = generateNewVarName(varToUpdate.getVarName(), 0);
        }

        if (newVarName == null || varToUpdate.getVarName().equals(newVarName)) {
            return varToUpdate;
        }

        val oldVarName = varToUpdate.getVarName();
        varToUpdate.setVarName(newVarName);
        updateVariableName(oldVarName, newVarName);
        return varToUpdate;
    }


    
    public SDVariable[] updateVariableNamesAndReferences(SDVariable[] variablesToUpdate, String[] newVariableNames) {

        int numVariables = variablesToUpdate.length;
        SDVariable[] updatedVariables = new SDVariable[numVariables];

        for (int i = 0; i < numVariables; i++) {
            SDVariable varToUpdate = variablesToUpdate[i];
            String name = newVariableNames == null ? null : newVariableNames[i];
            updatedVariables[i] = updateVariableNameAndReference(varToUpdate, name);
        }

        return updatedVariables;
    }

    



    private SDVariable[] outputs;
    private SDVariable[] inputs;


    private Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec_cache;

    public Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> exec() {

        

        if (log.isTraceEnabled()) {
            log.trace("Starting execution: {} functions", functionInstancesById.size());
        }


        if (!resolvedVariables)
            resolveVariablesWith(new LinkedHashMap<String, INDArray>());

        List<DifferentialFunction> ops = new ArrayList<>();


        localFlowPath.set(new FlowPath());

        val flowPath = localFlowPath.get();

        Map<SDVariable, DifferentialFunction> opMap = new HashMap<>();
        val funcs = new ArrayList<DifferentialFunction>(functionInstancesById.values());
        boolean onBackward = false;


        val frames = new ArrayDeque<String>();


        boolean inFrame = false;


        boolean frameLeft = false;

        int i = 0;
        int exec_counter = 0;
        for (; i < funcs.size(); i++) {
            ++exec_counter;

            if (log.isTraceEnabled()) {
                val f = funcs.get(i);
                String[] argNames = f.argNames();
                String[] outNames = f.outputVariablesNames();
                log.trace("Starting execution of step {} of {}: Function {} (ownName={}) - {}", exec_counter, funcs.size(),
                        f.opName(), f.getOwnName(), f.getClass().getName());
                log.trace("Function inputs: {} - Function outputs: {}", (argNames == null ? "(none)" : Arrays.toString(argNames)),
                        (outNames == null ? "(none)" : Arrays.toString(outNames)));
                SDVariable[] args = f.args();
                for (int arg = 0; arg < args.length; arg++) {
                    if (args[arg] == null) {
                        log.trace("--> arg {} - {}: argument is null!", arg, argNames[arg]);
                    } else {
                        INDArray arr = args[arg].getArr();
                        String arrShape = (arr == null ? "<array not present>" : Arrays.toString(arr.shape()));
                        log.trace("--> arg {} - {}: array shape: {}", arg, argNames[arg], arrShape);
                    }

                }
            }

            val opName = funcs.get(i).opName();
            if (!onBackward && GradientBackwardsMarker.OP_NAME.equals(opName)) {
                onBackward = true;
            }

            if (GradientBackwardsMarker.OP_NAME.equals(opName))
                continue;

            DifferentialFunction differentialFunction = funcs.get(i);
            val ownName = differentialFunction.getOwnName();


            flowPath.ensureNodeStateExists(differentialFunction.getOwnName());

            if (differentialFunction instanceof SDVariable) {
                if (log.isTraceEnabled()) {
                    log.trace("Skipping differentialFunction that is instanceof SDVariable: {}", opName);
                }
                continue;
            }

            val args = getInputsForFunction(differentialFunction);

            log.debug("Step: {}; Executing op {} for node [{}]", exec_counter, opName, ownName);



            boolean shouldSkip = false;
            if (differentialFunction instanceof Merge) {
                val arg0 = args[0];
                val arg1 = args[1];

                if (!flowPath.isActive(arg0) && !flowPath.isActive(arg1))
                    shouldSkip = true;
            } else {
                if (!(differentialFunction instanceof Exit)) {


                    if (frameLeft) {
                        frameLeft = false;

                        val frame_name = frames.removeLast();
                        flowPath.activateFrame(frame_name, false);
                        flowPath.forgetFrame(frame_name);
                    }


                    for (val input : args) {
                        if (!flowPath.isActive(input)) {

                            flowPath.markActive(differentialFunction.getOwnName(), false);
                            shouldSkip = true;
                            break;
                        }
                    }
                }
            }

            if (shouldSkip) {
                if (log.isTraceEnabled()) {
                    log.trace("Skipping function {}: shouldSkip = true", opName);
                }
                continue;
            }

            differentialFunction.resolvePropertiesFromSameDiffBeforeExecution();
            flowPath.markActive(differentialFunction.getOwnName(), true);

            
            if (differentialFunction instanceof LoopCond) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of LoopCond op");


                val inputs = getInputVariablesForFunction(differentialFunction);

                val array = inputs[0].getArr();
                variableNameToArr.put(differentialFunction.getOwnName(), array.dup(array.ordering()));

                flowPath.markExecuted(differentialFunction.getOwnName(), true);

                if ((int) array.getDouble(0) == 1) {
                    val frameName = frames.getLast();

                    flowPath.incrementNumberOfCycles(frameName);
                }
            } else if (differentialFunction instanceof Enter) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of Enter op");




                val inputs = getInputVariablesForFunction(differentialFunction);

                val array = inputs[0].getArr();
                val name = inputs[0].getVarName();

                if (array != null)
                    variableNameToArr.put(differentialFunction.getOwnName(), array.dup(array.ordering()));

                flowPath.markExecuted(differentialFunction.getOwnName(), true);


                val frame_name = ((Enter) differentialFunction).getFrameName();
                if (!flowPath.isRegisteredFrame(frame_name)) {
                    flowPath.registerFrame(frame_name);
                    frames.addLast(frame_name);
                    inFrame = true;
                }


            } else if (differentialFunction instanceof Exit) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of Exit op");



                val frame_name = frames.getLast();


                ((Exit) differentialFunction).setFrameName(frame_name);

                if (!flowPath.isFrameActive(frame_name)) {
                    flowPath.markActive(differentialFunction.getOwnName(), false);


                    frameLeft = true;
                    continue;
                }




                if (flowPath.isRewindPlanned(frame_name)) {

                    flowPath.planRewind(frame_name, false);
                    val currentPosition = i;
                    i = flowPath.getRewindPosition(frame_name);
                    val startPosition = i + 1;
                    flowPath.setRewindPosition(frame_name, -1);

                    continue;
                }

                val inputs = getInputVariablesForFunction(differentialFunction);

                val array = inputs[0].getArr();
                variableNameToArr.put(differentialFunction.getOwnName(), array.dup(array.ordering()));

                flowPath.markExecuted(differentialFunction.getOwnName(), true);


                frameLeft = true;

            } else if (differentialFunction instanceof NextIteration) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of NextIteration op");


                val inputs = getInputVariablesForFunction(differentialFunction);
                val frame_name = frames.getLast();

                val array = inputs[0].getArr();
                variableNameToArr.put(differentialFunction.getOwnName(), array.dup(array.ordering()));

                flowPath.markExecuted(differentialFunction.getOwnName(), true);


                if (!flowPath.isRewindPlanned(frame_name)) {
                    flowPath.planRewind(frame_name, true);

                    continue;
                }

            } else if (differentialFunction instanceof Merge) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of Merge op");



                val inputs = getInputVariablesForFunction(differentialFunction);

                val frame_name = frames.size() > 0 ? frames.getLast() : null;

                if (frame_name != null)
                    flowPath.activateFrame(frame_name, true);


                if (frame_name != null)
                    flowPath.setRewindPositionOnce(frame_name, i - 1);


                if (inputs.length == 2) {
                    val secondArg = functionInstancesById.get(inputs[1].getVarName());

                    if (secondArg != null && secondArg instanceof NextIteration) {
                        ((NextIteration) secondArg).setFrameName(frame_name);
                    }
                }


                if (flowPath.wasExecuted(inputs[1].getVarName())) {

                    val array = inputs[1].getArr();

                    if (array != null)
                        variableNameToArr.put(differentialFunction.getOwnName(), array.dup(array.ordering()));


                    flowPath.markExecuted(inputs[1].getVarName(), false);
                } else {

                    val array = inputs[0].getArr();

                    if (array != null)
                        variableNameToArr.put(differentialFunction.getOwnName(), array.dup(array.ordering()));
                }

                flowPath.markExecuted(differentialFunction.getOwnName(), true);
            } else if (differentialFunction instanceof Switch) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of Switch op");


                ((CustomOp) differentialFunction).populateInputsAndOutputsFromSameDiff();

                val inputs = getInputVariablesForFunction(differentialFunction);

                val input = inputs[0].getArr();
                val bool = inputs[1].getArr();


                if ((int) bool.getDouble(0) == 0) {

                    flowPath.setActiveBranch(differentialFunction.getOwnName(), 0);
                    flowPath.markActive(differentialFunction.getOwnName(), true);
                    flowPath.markActive(differentialFunction.getOwnName() + ":1", false);

                    if (input != null)
                        variableNameToArr.put(differentialFunction.getOwnName(), input.dup(input.ordering()));
                } else {

                    flowPath.setActiveBranch(differentialFunction.getOwnName(), 1);

                    if (input != null)
                        variableNameToArr.put(differentialFunction.getOwnName() + ":1", input.dup(input.ordering()));

                    flowPath.markActive(differentialFunction.getOwnName(), false);
                    flowPath.markActive(differentialFunction.getOwnName() + ":1", true);
                }

                flowPath.markExecuted(differentialFunction.getOwnName(), true);
            } else if (differentialFunction instanceof BaseTensorOp) {

                log.info("Starting execution of Tensor op [{}]", opName);


                val list = ((BaseTensorOp) differentialFunction).execute(this);

                if (!lists.containsKey(list.getName()))
                    lists.put(list.getName(), list);

                ops.add(differentialFunction);
            } else if (differentialFunction instanceof If) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of If op");

                If ifOp = (If) differentialFunction;
                if (!onBackward) {
                    ifOp.getPredicateExecution().exec();


                    if (ifOp.getTargetBoolean().getArr().sumNumber().doubleValue() > 0) {
                        ifOp.getLoopBodyExecution().exec();
                        ifOp.exectedTrueOrFalse(true);
                    } else {
                        ifOp.getFalseBodyExecution().exec();
                        ifOp.exectedTrueOrFalse(false);

                    }
                } else {
                    if (ifOp.getTrueBodyExecuted() != null) {
                        Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> execBackwards = null;
                        List<SDVariable> variablesForFunctions = null;
                        if (ifOp.getTrueBodyExecuted()) {
                            execBackwards = ifOp.getLoopBodyExecution().execBackwards();

                            variablesForFunctions = ifOp.getLoopBodyExecution().getVariablesAssociatedWithFunctions(execBackwards.getRight());
                        } else {
                            execBackwards = ifOp.getFalseBodyExecution().execBackwards();
                            variablesForFunctions = ifOp.getFalseBodyExecution().getVariablesAssociatedWithFunctions(execBackwards.getRight());
                        }

                        
                        for (SDVariable variable : variablesForFunctions) {
                            SDVariable proxyVar = var(variable);
                        }


                    } else
                        throw new ND4JIllegalStateException("No body was run.");

                }

                flowPath.markExecuted(differentialFunction.getOwnName(), true);

                ops.add(differentialFunction);

            } else if (differentialFunction instanceof While) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of While op");

                While whileOp = (While) differentialFunction;

                if (!onBackward) {
                    SameDiff execBody = whileOp.getLoopBodyExecution();





                    whileOp.getPredicateExecution().exec();
                    if (execBody.outputs == null) {



                        while (whileOp.getTargetBoolean().getArr().sumNumber().doubleValue() > 0) {

                            execBody.exec();
                            whileOp.getPredicateExecution().exec();
                            whileOp.incrementLoopCounter();
                        }
                    } else {
                        if (whileOp.getTargetBoolean().getSameDiff().inputs == null) {
                            whileOp.getTargetBoolean().getSameDiff().inputs = new SDVariable[whileOp.getInputVars().length];
                            for (int e = 0; e < whileOp.getInputVars().length; e++) {
                                whileOp.getTargetBoolean().getSameDiff().inputs[i] = whileOp.getTargetBoolean().getSameDiff().variables().get(i);
                            }
                        }
                        while (whileOp.getTargetBoolean().getArr().sumNumber().doubleValue() > 0) {

                            execBody.exec();
                            val outputs = execBody.outputs;

                            int cnt = 0;
                            for (val out : execBody.outputs) {
                                execBody.associateArrayWithVariable(out.getArr(), execBody.inputs[cnt]);
                                whileOp.getTargetBoolean().getSameDiff().associateArrayWithVariable(out.getArr(),
                                        whileOp.getTargetBoolean().getSameDiff().inputs[cnt++]);
                            }

                            whileOp.getPredicateExecution().exec();
                            whileOp.incrementLoopCounter();

                        }
                    }

                    List<SDVariable> outputs = new ArrayList<>();
                    val outputFuncArgs = new ArrayList<>(execBody.functionInstancesById.values()).get(execBody.functionInstancesById.values().size() - 1).outputVariables();
                    outputs.addAll(Arrays.asList(outputFuncArgs));

                    whileOp.setOutputVars(outputs.toArray(new SDVariable[outputs.size()]));
                    ops.add(differentialFunction);
                } else {
                    
                    Pair<Map<SDVariable, DifferentialFunction>, List<DifferentialFunction>> mapListPair = whileOp.getLoopBodyExecution().execBackwards();
                    for (SDVariable variable : mapListPair.getFirst().keySet()) {
                        variable.getArr().muli(whileOp.getNumLooped());
                    }


                }

                flowPath.markExecuted(differentialFunction.getOwnName(), true);

            } else if (differentialFunction instanceof CustomOp) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of CustomOp op");

                DynamicCustomOp customOp = (DynamicCustomOp) differentialFunction;
                try {
                    customOp.populateInputsAndOutputsFromSameDiff();
                } catch (Throwable t) {
                    throw new RuntimeException("Error populating inputs and outputs for function \"" + differentialFunction.getOwnName()
                            + "\" of type " + differentialFunction.getClass().getName(), t);
                }
                customOp.assertValidForExecution();

                customOp.updateInputsFromSameDiff();

                Nd4j.getExecutioner().exec(customOp);

                

                flowPath.markExecuted(differentialFunction.getOwnName(), true);

                ops.add(customOp);
            } else if (differentialFunction instanceof Op) {
                if (log.isTraceEnabled())
                    log.trace("Starting execution of Op op");

                val inputs = getInputVariablesForFunction(differentialFunction);

                Op op = (Op) differentialFunction;


                op.setX(inputs[0].getArr());
                if (inputs.length == 2)
                    op.setY(inputs[1].getArr());

                if (differentialFunction.getDimensions() == null)
                    Nd4j.getExecutioner().exec(op);
                else if (op.isExecSpecial()) {
                    op.exec();
                } else {
                    int[] axes = differentialFunction.getDimensions();
                    if (differentialFunction instanceof Accumulation) {
                        Accumulation accumulation = (Accumulation) differentialFunction;

                        Nd4j.getExecutioner().exec(accumulation, axes);

                        if (differentialFunction.outputVariables()[0].getArr() == null) {
                            val var = differentialFunction.outputVariables()[0];
                            updateVariable(var.getVarName(), accumulation.z());
                            updateShapeForVarName(var.getVarName(), accumulation.z().shape());
                        }
                    } else if (differentialFunction instanceof BroadcastOp) {
                        BroadcastOp broadcastOp = (BroadcastOp) differentialFunction;
                        Nd4j.getExecutioner().exec(broadcastOp, axes);
                    } else if (differentialFunction instanceof GradientOp) {
                        Nd4j.getExecutioner().exec(op);
                    } else if (differentialFunction instanceof IndexAccumulation) {
                        IndexAccumulation indexAccumulation = (IndexAccumulation) differentialFunction;
                        Nd4j.getExecutioner().exec(indexAccumulation, axes);

                    } else if (differentialFunction instanceof TransformOp) {
                        TransformOp t = (TransformOp) differentialFunction;
                        Nd4j.getExecutioner().exec(t, axes);
                    }
                }


                flowPath.markExecuted(differentialFunction.getOwnName(), true);

                ops.add(differentialFunction);
            } else {
                throw new IllegalStateException("Unknown function type: " + differentialFunction.getClass().getName());
            }




            if (log.isTraceEnabled()) {
                log.trace("Execution completed for DifferentialFunction {} - {}", opName, differentialFunction.getOwnName());
                SDVariable[] outputVars = differentialFunction.outputVariables();
                for (int x = 0; x < outputVars.length; x++) {
                    INDArray arr = outputVars[x].getArr();
                    String arrShape = (arr == null ? "<no array>" : Arrays.toString(arr.shape()));
                    log.trace("--> output {} - {}: array shape {}", x, outputVars[x].getVarName(), arrShape);
                }
            }
        }

        if (log.isTraceEnabled()) {
            log.trace("Execution complete");
        }

        val ret = new Pair<>(opMap, ops);
        exec_cache = ret;
        if (parent != null) {
            parent.exec_cache = exec_cache;
        }


        return ret;
    }


    
    public void printFunction(DifferentialFunction differentialFunction) {
        if (!logExecution)
            return;
        if (differentialFunction instanceof SDVariable)
            return;

        StringBuilder argShapes = new StringBuilder();
        for (val arg : differentialFunction.args()) {
            argShapes.append(" Variable " + arg.getVarName() +
                    " Shape for " + Arrays.toString(arg.getShape()));
        }

        for (val func : differentialFunction.outputVariables()) {
            argShapes.append("  Output variable " + func.getVarName() + " is " +
                    Arrays.toString(func.getShape()));
        }


        StringBuilder realShapes = new StringBuilder();
        for (val arg : differentialFunction.args()) {
            realShapes.append(" Input shape for " + arg.getVarName() + " is  " + Arrays.
                    toString(getShapeForVarName(arg.getVarName())));
        }

        for (val arg : differentialFunction.outputVariables()) {
            realShapes.append(" Output shape for " + arg.getVarName() + " is  " + Arrays.
                    toString(getShapeForVarName(arg.getVarName())));
        }



    }


    
    public static int[] permuteDataFormatForSameDiff(String dataFormat, boolean weights) {
        val dl4jFormat = "NCHW";
        dataFormat = dataFormat.toUpperCase();

        


        


        int[] ret = new int[4];
        if (weights) {
            ret[0] = dataFormat.indexOf('W');
            ret[1] = dataFormat.indexOf('C');
            ret[2] = dataFormat.indexOf('N');
            ret[3] = dataFormat.indexOf('H');
            return ret;
        }




        for (int i = 0; i < dataFormat.length(); i++) {
            if (dl4jFormat.indexOf(dataFormat.charAt(i)) < 0) {
                throw new ND4JIllegalStateException("Illegal convolution data format string passed in " + dataFormat + " must be some variant of NCHW");
            }
        }

        for (int i = 0; i < dl4jFormat.length(); i++) {
            ret[i] = dl4jFormat.indexOf(dataFormat.charAt(i));
        }

        return ret;
    }

    
    public void updateVariable(String variableName, INDArray arr) {
        if (!variableNameToArr.containsKey(variableName))
            putArrayForVarName(variableName, arr);
        else
            updateArrayForVarName(variableName, arr);
    }


    protected int asFlatNode(String name, @NonNull SameDiff scope, @NonNull FlatBufferBuilder bufferBuilder) {
        int scopeName = bufferBuilder.createString(name);

        int flatNode = FlatNode.createFlatNode(bufferBuilder,
                scopeName,
                scopeName,
                OpType.LOGIC,
                10, 
                0,
                0,
                0,
                (byte) 0,
                0,
                0,
                0,
                0,
                -1,
                0.0f, 0, 0);

        return flatNode;
    }

    
    public static Pair<String, Integer> parseVariable(@NonNull String varName) {
        if (!varName.contains(":")) {
            return Pair.pairOf(varName, 0);
        } else {
            val split = varName.split(":");
            val index = Integer.valueOf(split[split.length - 1]);
            if (split.length == 2)
                return Pair.pairOf(split[0], index);
            else {
                val builder = new StringBuilder();
                for (int e = 0; e < split.length - 1; e++) {
                    builder.append(split[e]);

                    if (e < split.length - 2)
                        builder.append(":");
                }

                return Pair.pairOf(builder.toString(), index);
            }
        }
    }

    protected int asFlatNode(@NonNull DifferentialFunction node, @NonNull FlatBufferBuilder bufferBuilder, List<SDVariable> variables, Map<String, Integer> reverseMap, Map<String, Integer> forwardMap, Map<String, Integer> framesMap, AtomicInteger idCounter) {
        val opName = node.opName();
        val hash = getOpNum(node.opName(), node.opType());


        double[] extras = node.getExtraArgs() != null ? new double[node.getExtraArgs().length] : new double[0];
        for (int e = 0; e < extras.length; e++) {
            extras[e] = ((Number) node.getExtraArgs()[e]).doubleValue();
        }

        long[] extraBits = null;
        if (node.opType() == Op.Type.CUSTOM) {
            DynamicCustomOp dynamicCustomOp = (DynamicCustomOp) node;
            extraBits = dynamicCustomOp.iArgs();
        } else if (node instanceof Enter) {

            val frameName = ((Enter) node).getFrameName();
            if (!framesMap.containsKey(frameName))
                framesMap.put(frameName, idCounter.incrementAndGet());

            extraBits = new long[]{framesMap.get(frameName).intValue()};
        } else
            extraBits = new long[]{};

        val inPaired = new ArrayList<Integer>();

        int[] outputIds = null;
        SDVariable[] outputVertexId = null;

        try {
            outputVertexId = node.outputVariables();
            outputIds = new int[outputVertexId.length];
            for (int i = 0; i < outputIds.length; i++) {
                outputIds[i] = variables.indexOf(outputVertexId[i]);
            }
        } catch (ND4UnresolvedOutputVariables e) {

            outputIds = new int[0];
            outputVertexId = null;
        } catch (Exception e) {
            throw new ND4JIllegalStateException(e);
        }


        val inputs = node.args();
        log.trace("");
        for (val input : inputs) {

            val pair = parseVariable(input.getVarName());
            if (!reverseMap.containsKey(pair.getFirst())) {
                if (pair.getFirst().contains("NextIteration")) {

                    int fwdNodeId = idCounter.incrementAndGet();
                    forwardMap.put(pair.getFirst(), fwdNodeId);
                    reverseMap.put(pair.getFirst(), fwdNodeId);
                } else {
                    throw new ND4JIllegalStateException("Unknown variable used in input: [" + pair.getFirst() + "]");
                }
            }

            int nodeId = reverseMap.get(pair.getFirst());
            int outputIndex = pair.getSecond();

            inPaired.add(IntPair.createIntPair(bufferBuilder, nodeId, outputIndex));

        }

        log.debug("Own Name: {}", node.getOwnName());
        int ownId = forwardMap.containsKey(node.getOwnName()) ? forwardMap.get(node.getOwnName()) : idCounter.incrementAndGet();
        reverseMap.put(node.getOwnName(), ownId);

        val dims = node.opType() == Op.Type.REDUCE && inPaired.size() == 1 && node.getDimensions() != null ? node.getDimensions() : new int[]{};
        
        List<FunctionProperties> props = new ArrayList<>();
        int properties = FunctionProperties.asFlatProperties(bufferBuilder, props);

        int nodesIn = FlatNode.createInputVector(bufferBuilder, new int[]{});
        int nodesInPaired = FlatNode.createInputPairedVector(bufferBuilder, Ints.toArray(inPaired));
        int nodesOut = FlatNode.createOutputVector(bufferBuilder, outputIds);
        int extraz = FlatNode.createExtraParamsVector(bufferBuilder, extras);
        int integerArgs = FlatNode.createExtraIntegerVector(bufferBuilder, extraBits);
        int dimensions = FlatNode.createDimensionsVector(bufferBuilder, dims);
        int fname = bufferBuilder.createString(
                outputVertexId == null ||
                        outputVertexId.length < 1 ||
                        outputVertexId[0] == null ? "" :
                        outputVertexId[0].getVarName());
        int scopeName = bufferBuilder.createString("");

        if (node.opType() == null)
            log.warn("Null-op node: {}", node);

        int flatNode = FlatNode.createFlatNode(
                bufferBuilder,
                ownId,
                fname,
                getFlatOpType(node.opType()),
                hash,
                properties,
                nodesIn,
                nodesInPaired,
                (byte) 0,
                nodesOut,
                extraz,
                integerArgs,
                dimensions,
                -1,
                node.opType() == Op.Type.SCALAR && node.getScalarValue() != null ? node.getScalarValue().floatValue() : 0.0f, 0, scopeName);

        return flatNode;
    }


    
    public ByteBuffer asFlatBuffers(@NonNull ExecutorConfiguration configuration) {
        Nd4j.getExecutioner().commit();
        FlatBufferBuilder bufferBuilder = new FlatBufferBuilder(1024);
        val idCounter = new AtomicInteger(0);

        val flatVariables = new ArrayList<Integer>();
        val flatOffsets = new ArrayList<Integer>();
        val flatNodes = new ArrayList<Integer>();


        List<SDVariable> variableList = new ArrayList<>(variables());
        val reverseMap = new LinkedHashMap<String, Integer>();
        val forwardMap = new LinkedHashMap<String, Integer>();
        val framesMap = new LinkedHashMap<String, Integer>();

        int idx = 0;
        for (val variable : variables()) {
            log.debug("Exporting variable: [{}]", variable.getVarName());
            if (variable.getArr() == null || variable.getShape() == null) {


                continue;
            }


            val pair = parseVariable(variable.getVarName());
            reverseMap.put(pair.getFirst(), idCounter.incrementAndGet());
            log.debug("Adding [{}] as [{}]", pair.getFirst(), idCounter.get());

            val arr = variable.getArr();

            int name = bufferBuilder.createString(variable.getVarName());
            int array = arr.toFlatArray(bufferBuilder);
            int id = IntPair.createIntPair(bufferBuilder, idCounter.get(), 0);


            int flatVariable = FlatVariable.createFlatVariable(bufferBuilder, id, name, 0, array, -1);
            flatVariables.add(flatVariable);
        }


        for (val func : functionInstancesById.values()) {
            flatNodes.add(asFlatNode(func, bufferBuilder, variableList, reverseMap, forwardMap, framesMap, idCounter));
        }


        for (val scope : sameDiffFunctionInstances.entrySet()) {
            flatNodes.add(asFlatNode(scope.getKey(), scope.getValue(), bufferBuilder));
            val currVarList = new ArrayList<SDVariable>(scope.getValue().variables());

            for (val node : scope.getValue().variables()) {
                INDArray arr = node.getArr();
                if (arr == null) {




                    continue;
                }

                int name = bufferBuilder.createString(node.getVarName());
                int array = arr.toFlatArray(bufferBuilder);
                int id = IntPair.createIntPair(bufferBuilder, ++idx, 0);

                val pair = parseVariable(node.getVarName());
                reverseMap.put(pair.getFirst(), idx);

                log.debug("Adding [{}] as [{}]", pair.getFirst(), idx);

                int flatVariable = FlatVariable.createFlatVariable(bufferBuilder, id, name, 0, array, -1);
                flatVariables.add(flatVariable);
            }


            for (val func : scope.getValue().functionInstancesById.values()) {
                flatNodes.add(asFlatNode(func, bufferBuilder, currVarList, reverseMap, forwardMap, framesMap, idCounter));
            }

        }

        int outputsOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(flatOffsets));
        int variablesOffset = FlatGraph.createVariablesVector(bufferBuilder, Ints.toArray(flatVariables));
        int nodesOffset = FlatGraph.createNodesVector(bufferBuilder, Ints.toArray(flatNodes));

        int fg = FlatGraph.createFlatGraph(bufferBuilder, 119, variablesOffset, nodesOffset, outputsOffset, configuration.getFlatConfiguration(bufferBuilder));
        bufferBuilder.finish(fg);

        synchronized (this) {
            if (this.reverseMap == null)
                this.reverseMap = reverseMap;
        }

        return bufferBuilder.dataBuffer();
    }

    
    public ByteBuffer asFlatBuffers() {
        val configuration = ExecutorConfiguration.builder()
                .outputMode(OutputMode.VARIABLE_SPACE)
                .executionMode(org.nd4j.autodiff.execution.conf.ExecutionMode.SEQUENTIAL)
                .profilingMode(OpExecutioner.ProfilingMode.DISABLED)
                .gatherTimings(true)
                .build();

        return asFlatBuffers(configuration);
    }

    
    public static ByteOrder getOrderFromByte(byte val) {
        if (val == org.nd4j.graph.ByteOrder.LE)
            return ByteOrder.LITTLE_ENDIAN;
        else
            return ByteOrder.BIG_ENDIAN;
    }

    
    public static byte getOrderAsByte() {
        if (ByteOrder.nativeOrder().equals(ByteOrder.BIG_ENDIAN))
            return org.nd4j.graph.ByteOrder.BE;
        else
            return org.nd4j.graph.ByteOrder.LE;
    }

    
    public void asFlatFile(@NonNull File file) throws IOException {
        val fb = asFlatBuffers();
        val offset = fb.position();

        val array = fb.array();

        try (val fos = new FileOutputStream(file); val bos = new BufferedOutputStream(fos); val dos = new DataOutputStream(bos)) {
            dos.write(array, offset, array.length - offset);
        }
    }

    
    public void asFlatFile(@NonNull File file, @NonNull ExecutorConfiguration configuration) throws IOException {
        val fb = asFlatBuffers(configuration);
        val offset = fb.position();

        val array = fb.array();

        try (val fos = new FileOutputStream(file); val bos = new BufferedOutputStream(fos); val dos = new DataOutputStream(bos)) {
            dos.write(array, offset, array.length - offset);
        }
    }

    
    public String asFlatPrint() {
        val sb = new StringBuilder();
        val fb = asFlatBuffers();

        val graph = FlatGraph.getRootAsFlatGraph(fb);

        sb.append("\nExternal variables:\n\n");
        for (int e = 0; e < graph.variablesLength(); e++) {
            val var = graph.variables(e);
            val ndarray = Nd4j.createFromFlatArray(var.ndarray());

            sb.append(var.id().first())
                    .append(":<").append(var.name()).append("> ")
                    .append(Arrays.toString(ndarray.shapeInfoDataBuffer().asInt())).append("; Values: ").append(Arrays.toString(ndarray.data().asFloat())).append(";\n");
        }

        val map = Nd4j.getExecutioner().getCustomOperations();


        sb.append("\nOps sequence:\n\n");
        for (int e = 0; e < graph.nodesLength(); e++) {
            val node = graph.nodes(e);

            log.info("{}:<{}>", node.id(), node.name());
            sb.append(node.id())
                    .append(":<").append(node.name()).append("> ").append(SameDiff.getTypeFromByte(node.opType()));

            if (SameDiff.getTypeFromByte(node.opType()) != Op.Type.CUSTOM)
                sb.append(": ").append(node.opNum());
            else {
                val keys = map.keySet();
                String opName = null;
                for (val k : keys) {
                    val d = map.get(k);
                    if (d.getHash() == node.opNum())
                        opName = k;
                }

                if (opName == null)
                    opName = "unknown";

                sb.append(": ").append(opName);
            }

            sb.append("; Inputs: {");

            for (int i = 0; i < node.inputPairedLength(); i++) {
                val pair = node.inputPaired(i);

                sb.append("[").append(pair.first()).append(":").append(pair.second()).append("]");

                if (i < node.inputPairedLength() - 1)
                    sb.append(", ");
            }

            sb.append("};");
            sb.append(" OpNum: {").append(node.opNum()).append("};");

            sb.append("\n");
        }


        return sb.toString();
    }

    
    public static DataBuffer.Type getDataTypeFromByte(byte val) {
        if (val == DataType.FLOAT)
            return DataBuffer.Type.FLOAT;
        else if (val == DataType.DOUBLE)
            return DataBuffer.Type.DOUBLE;
        else if (val == DataType.HALF)
            return DataBuffer.Type.HALF;

        throw new UnsupportedOperationException("Unsupported DataType: [" + val + "]");
    }

    
    public static byte getDataTypeAsByte(DataBuffer.Type type) {
        switch (type) {
            case FLOAT:
                return DataType.FLOAT;
            case DOUBLE:
                return DataType.DOUBLE;
            case HALF:
                return DataType.HALF;
            case INT:
                return DataType.INT32;
            case LONG:
                return DataType.INT64;
            default:
                throw new ND4JIllegalStateException("Unknown or unsupported DataType used: [" + type + "]");
        }
    }

    
    public static long getOpNum(String name, Op.Type type) {
        if (type == Op.Type.LOOP) {
            return 0;
        } else if (type == Op.Type.RETURN) {
            return 40;
        } else if (type == Op.Type.IF) {
            return 30;
        } else if (type == Op.Type.CONDITIONAL) {
            return 10;
        } else if (type == Op.Type.MERGE) {
            return 60L;
        } else if (type == Op.Type.LOOP_COND) {
            return 70L;
        } else if (type == Op.Type.NEXT_ITERATION) {
            return 80L;
        } else if (type == Op.Type.EXIT) {
            return 90L;
        } else if (type == Op.Type.ENTER) {
            return 100L;
        } else if (type == Op.Type.CUSTOM) {
            val name2 = Nd4j.getExecutioner().getCustomOperations().get(name.toLowerCase());
            if (name2 == null)
                return 0;
            return Nd4j.getExecutioner().getCustomOperations().get(name.toLowerCase()).getHash();

        } else
            return (long) Nd4j.getOpFactory().getOpNumByName(name);
    }

    
    public static Op.Type getTypeFromByte(byte type) {
        switch (type) {
            case OpType.SCALAR:
                return Op.Type.SCALAR;
            case OpType.BROADCAST:
                return Op.Type.BROADCAST;
            case OpType.TRANSFORM:
                return Op.Type.TRANSFORM;
            case OpType.ACCUMULATION:
                return Op.Type.REDUCE;
            case OpType.ACCUMULATION3:
                return Op.Type.REDUCE3;
            case OpType.INDEX_ACCUMULATION:
                return Op.Type.INDEXREDUCE;
            case OpType.RANDOM:
                return Op.Type.RANDOM;
            case OpType.LOGIC:
                return Op.Type.META;
            case OpType.CUSTOM:
                return Op.Type.CUSTOM;
            case OpType.SHAPE:
                return Op.Type.SHAPE;
            case OpType.PAIRWISE:
                return Op.Type.PAIRWISE;
            case OpType.SUMMARYSTATS:
                return Op.Type.SUMMARYSTATS;
            default:
                throw new UnsupportedOperationException("Unknown op type passed in: " + type);
        }
    }

    
    public static byte getFlatOpType(Op.Type type) {
        switch (type) {
            case SCALAR:
                return OpType.SCALAR;
            case BROADCAST:
                return OpType.BROADCAST;
            case TRANSFORM:
            case SPECIAL:
                return OpType.TRANSFORM;
            case REDUCE:
                return OpType.ACCUMULATION;
            case REDUCE3:
                return OpType.ACCUMULATION3;
            case INDEXREDUCE:
                return OpType.INDEX_ACCUMULATION;
            case RANDOM:
                return OpType.RANDOM;
            case MERGE:
            case CONDITIONAL:
            case LOOP:
            case RETURN:
            case ENTER:
            case EXIT:
            case NEXT_ITERATION:
            case LOOP_COND:
            case IF:
                return OpType.LOGIC;
            case CUSTOM:
                return OpType.CUSTOM;
            case SHAPE:
                return OpType.SHAPE;
            case PAIRWISE:
                return OpType.PAIRWISE;
            case SUMMARYSTATS:
                return OpType.SUMMARYSTATS;
            default:
                throw new UnsupportedOperationException("Unknown op type passed in: " + type);
        }
    }


    public String summary() {

        Map<String, SDVariable> varMap = variableMap();
        DifferentialFunction[] functions = functions();


        int countVarsWithArrays = 0;
        for (String s : varMap.keySet()) {
            if (getArrForVarName(s) != null) {
                countVarsWithArrays++;
            }
        }

        StringBuilder sb = new StringBuilder();
        String format = "%-25s%-20s";
        sb.append("--- Summary ---\n");
        sb.append(String.format(format, "Variables:", varMap.size())).append(" (").append(countVarsWithArrays).append(" with arrays)").append("\n")
                .append(String.format(format, "Functions:", functions.length)).append("\n")
                .append(String.format(format, "SameDiff Function Defs:", sameDiffFunctionInstances.size()))
                .append("\n\n");

        sb.append("--- Variables ---\n");

        Map<String, String> outputOfFn = new HashMap<>();
        int maxLengthOutputOf = 22;     
        for (String s : varMap.keySet()) {
            String outputOf = null;
            for (Map.Entry<String, String[]> dfToArgs : outgoingArgsReverse.entrySet()) {
                if (dfToArgs.getValue() != null && ArrayUtils.contains(dfToArgs.getValue(), s)) {
                    outputOf = dfToArgs.getKey();
                    break;
                }
            }

            if (outputOf == null) {
                outputOf = "<none>";
            } else {
                DifferentialFunction d = getFunctionById(outputOf);
                outputOf = d.getOwnName() + "(" + d.opName() + ")";
            }
            outputOfFn.put(s, outputOf);
            maxLengthOutputOf = Math.max(maxLengthOutputOf, outputOf.length());
        }
        maxLengthOutputOf += 2;


        format = "%-20s%-20s%-" + maxLengthOutputOf + "s%-20s";
        sb.append(String.format(format, "- Name -", "- Array Shape -", "- Output Of Function -", "- Inputs To Functions -")).append("\n");
        for (String s : varMap.keySet()) {
            INDArray arr = getArrForVarName(s);
            String arrayShape = "-";
            if (arr != null) {
                arrayShape = Arrays.toString(arr.shape());
            }

            List<DifferentialFunction> dfs = functionsArgsFor.get(s);
            String dfArrStr = "";
            if (dfs != null) {
                String[] dfArr = new String[dfs.size()];
                for (int i = 0; i < dfs.size(); i++) {
                    dfArr[i] = dfs.get(i).getOwnName();
                }
                dfArrStr = Arrays.toString(dfArr);
            }

            String outputOfStr = outputOfFn.get(s);

            sb.append(String.format(format, s, arrayShape, outputOfStr, dfArrStr)).append("\n");
        }

        sb.append("\n\n--- Functions ---\n");


        List<String> dfInputStr = new ArrayList<>();
        List<String> dfOutputStr = new ArrayList<>();
        int maxInLength = 10;       
        int maxOutLength = 11;      
        int maxOpNameLength = 10;   
        int maxDfClassNameLength = 10;  
        for (DifferentialFunction df : functions) {
            SDVariable[] args = df.args();
            SDVariable[] outputs = df.outputVariables();

            String[] argNames = df.argNames();
            String[] outNames = df.outputVariablesNames();

            String argStr = Arrays.toString(argNames);
            String outStr = Arrays.toString(outNames);

            maxInLength = Math.max(maxInLength, argStr.length());
            maxOutLength = Math.max(maxOutLength, outStr.length());

            dfInputStr.add(argStr);
            dfOutputStr.add(outStr);

            String name = df.getOwnName() == null ? df.opName() : df.getOwnName();
            maxOpNameLength = Math.max(maxOpNameLength, name.length());
            maxDfClassNameLength = Math.max(maxDfClassNameLength, df.getClass().getSimpleName().length());
        }

        maxInLength += 2;
        maxOutLength += 2;
        maxOpNameLength += 2;
        maxDfClassNameLength += 2;


        format = "%-5s%-" + maxOpNameLength + "s%-" + maxDfClassNameLength + "s%-" + maxInLength + "s%-" + maxOutLength + "s";
        sb.append(String.format(format, "", "- Function Name -", "- Op -", "- Inputs -", "- Outputs -")).append("\n");
        for (int i = 0; i < functions.length; i++) {
            DifferentialFunction df = functions[i];
            String fnName = df.getOwnName() == null ? df.opName() : df.getOwnName();

            sb.append(String.format(format, String.valueOf(i), fnName, df.getClass().getSimpleName(), dfInputStr.get(i), dfOutputStr.get(i))).append("\n");
        }

        if (sameDiffFunctionInstances.size() > 0) {
            sb.append("\n\n--- SameDiff Defined Functions ---\n");
            format = "%-20s%-15s%-15s%-15s";
            sb.append(String.format(format, "- Name -", "- Variables -", "- Functions -", "- Fn Defs -")).append("\n");
            for (Map.Entry<String, SameDiff> e : sameDiffFunctionInstances.entrySet()) {
                SameDiff sd = e.getValue();
                int vars = sd.variableMap().size();
                int fns = (sd.functions() == null ? 0 : sd.functions().length);
                int defFns = sd.definedFunctionNames().size();

                sb.append(String.format(format, e.getKey(), String.valueOf(vars), String.valueOf(fns), String.valueOf(defFns))).append("\n");
            }
        }

        return sb.toString();
    }
}

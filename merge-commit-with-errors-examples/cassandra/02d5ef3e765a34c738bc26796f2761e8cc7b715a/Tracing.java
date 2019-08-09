
package org.apache.cassandra.tracing;

import java.net.InetAddress;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.cassandra.concurrent.Stage;
import org.apache.cassandra.concurrent.StageManager;
import org.apache.cassandra.config.CFMetaData;
import org.apache.cassandra.cql3.ColumnNameBuilder;
import org.apache.cassandra.db.*;
import org.apache.cassandra.db.marshal.TimeUUIDType;
import org.apache.cassandra.exceptions.OverloadedException;
import org.apache.cassandra.exceptions.UnavailableException;
import org.apache.cassandra.exceptions.WriteTimeoutException;
import org.apache.cassandra.net.MessageIn;
import org.apache.cassandra.net.MessagingService;
import org.apache.cassandra.service.StorageProxy;
import org.apache.cassandra.utils.ByteBufferUtil;
import org.apache.cassandra.utils.FBUtilities;
import org.apache.cassandra.utils.UUIDGen;

import static org.apache.cassandra.utils.ByteBufferUtil.bytes;


public class Tracing
{
    public static final String TRACE_KS = "system_traces";
    public static final String EVENTS_CF = "events";
    public static final String SESSIONS_CF = "sessions";
    public static final String TRACE_HEADER = "TraceSession";

    private static final int TTL = 24 * 3600;

    private static final Logger logger = LoggerFactory.getLogger(Tracing.class);

    private final InetAddress localAddress = FBUtilities.getLocalAddress();

    private final ThreadLocal<TraceState> state = new ThreadLocal<TraceState>();

    private final ConcurrentMap<UUID, TraceState> sessions = new ConcurrentHashMap<UUID, TraceState>();

    public static final Tracing instance = new Tracing();

    public static void addColumn(ColumnFamily cf, ByteBuffer name, InetAddress address)
    {
        addColumn(cf, name, ByteBufferUtil.bytes(address));
    }

    public static void addColumn(ColumnFamily cf, ByteBuffer name, int value)
    {
        addColumn(cf, name, ByteBufferUtil.bytes(value));
    }

    public static void addColumn(ColumnFamily cf, ByteBuffer name, long value)
    {
        addColumn(cf, name, ByteBufferUtil.bytes(value));
    }

    public static void addColumn(ColumnFamily cf, ByteBuffer name, String value)
    {
        addColumn(cf, name, ByteBufferUtil.bytes(value));
    }

    private static void addColumn(ColumnFamily cf, ByteBuffer name, ByteBuffer value)
    {
        cf.addColumn(new ExpiringColumn(name, value, System.currentTimeMillis(), TTL));
    }

    public void addParameterColumns(ColumnFamily cf, Map<String, String> rawPayload)
    {
        for (Map.Entry<String, String> entry : rawPayload.entrySet())
        {
            cf.addColumn(new ExpiringColumn(buildName(cf.metadata(), bytes("parameters"), bytes(entry.getKey())),
                                            bytes(entry.getValue()), System.currentTimeMillis(), TTL));
        }
    }

    public static ByteBuffer buildName(CFMetaData meta, ByteBuffer... args)
    {
        ColumnNameBuilder builder = meta.getCfDef().getColumnNameBuilder();
        for (ByteBuffer arg : args)
            builder.add(arg);
        return builder.build();
    }

    public UUID getSessionId()
    {
        assert isTracing();
        return state.get().sessionId;
    }

    
    public static boolean isTracing()
    {
        return instance.state.get() != null;
    }

    public UUID newSession()
    {
        return newSession(TimeUUIDType.instance.compose(ByteBuffer.wrap(UUIDGen.getTimeUUIDBytes())));
    }

    public UUID newSession(UUID sessionId)
    {
        assert state.get() == null;

        TraceState ts = new TraceState(localAddress, sessionId);
        state.set(ts);
        sessions.put(sessionId, ts);

        return sessionId;
    }

    public void stopNonLocal(TraceState state)
    {
        sessions.remove(state.sessionId);
    }

    
    public void stopSession()
    {
        TraceState state = this.state.get();
        if (state == null) 
        {
            logger.debug("request complete");
        }
        else
        {
            final int elapsed = state.elapsed();
            final ByteBuffer sessionIdBytes = state.sessionIdBytes;

            StageManager.getStage(Stage.TRACING).execute(new Runnable()
            {
                public void run()
                {
                    CFMetaData cfMeta = CFMetaData.TraceSessionsCf;
                    ColumnFamily cf = ArrayBackedSortedColumns.factory.create(cfMeta);
                    addColumn(cf, buildName(cfMeta, bytes("duration")), elapsed);
<<<<<<< HEAD
                    RowMutation mutation = new RowMutation(TRACE_KS, sessionIdBytes, cf);
                    StorageProxy.mutate(Arrays.asList(mutation), ConsistencyLevel.ANY);
||||||| merged common ancestors
                    RowMutation mutation = new RowMutation(TRACE_KS, sessionIdBytes);
                    mutation.add(cf);
                    StorageProxy.mutate(Arrays.asList(mutation), ConsistencyLevel.ANY);
=======
                    mutateWithCatch(new RowMutation(TRACE_KS, sessionIdBytes, cf));
>>>>>>> cassandra-1.2
                }
            });

            sessions.remove(state.sessionId);
            this.state.set(null);
        }
    }

    public TraceState get()
    {
        return state.get();
    }

    public TraceState get(UUID sessionId)
    {
        return sessions.get(sessionId);
    }

    public void set(final TraceState tls)
    {
        state.set(tls);
    }

    public void begin(final String request, final Map<String, String> parameters)
    {
        assert isTracing();

        final long started_at = System.currentTimeMillis();
        final ByteBuffer sessionIdBytes = state.get().sessionIdBytes;

        StageManager.getStage(Stage.TRACING).execute(new Runnable()
        {
            public void run()
            {
                CFMetaData cfMeta = CFMetaData.TraceSessionsCf;
                ColumnFamily cf = TreeMapBackedSortedColumns.factory.create(cfMeta);
                addColumn(cf, buildName(cfMeta, bytes("coordinator")), FBUtilities.getBroadcastAddress());
                addParameterColumns(cf, parameters);
                addColumn(cf, buildName(cfMeta, bytes("request")), request);
                addColumn(cf, buildName(cfMeta, bytes("started_at")), started_at);
                mutateWithCatch(new RowMutation(TRACE_KS, sessionIdBytes, cf));
            }
        });
    }

    
    public TraceState initializeFromMessage(final MessageIn<?> message)
    {
        final byte[] sessionBytes = message.parameters.get(Tracing.TRACE_HEADER);

        if (sessionBytes == null)
            return null;

        assert sessionBytes.length == 16;
        UUID sessionId = UUIDGen.getUUID(ByteBuffer.wrap(sessionBytes));
        TraceState ts = sessions.get(sessionId);
        if (ts != null)
            return ts;

        if (message.verb == MessagingService.Verb.REQUEST_RESPONSE)
        {

            return new ExpiredTraceState(sessionId);
        }
        else
        {
            ts = new TraceState(message.from, sessionId);
            sessions.put(sessionId, ts);
            return ts;
        }
    }

    public static void trace(String message)
    {
        final TraceState state = instance.get();
        if (state == null) 
            return;

        state.trace(message);
    }

    public static void trace(String format, Object arg)
    {
        final TraceState state = instance.get();
        if (state == null) 
            return;

        state.trace(format, arg);
    }

    public static void trace(String format, Object arg1, Object arg2)
    {
        final TraceState state = instance.get();
        if (state == null) 
            return;

        state.trace(format, arg1, arg2);
    }

    public static void trace(String format, Object[] args)
    {
        final TraceState state = instance.get();
        if (state == null) 
            return;

        state.trace(format, args);
    }

    static void mutateWithCatch(RowMutation mutation)
    {
        try
        {
            StorageProxy.mutate(Arrays.asList(mutation), ConsistencyLevel.ANY);
        }
        catch (UnavailableException | WriteTimeoutException e)
        {

            throw new AssertionError(e);
        }
        catch (OverloadedException e)
        {
            logger.warn("Too many nodes are overloaded to save trace events");
        }
    }
}

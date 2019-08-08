

package io.realm;

import android.content.Context;

import java.io.File;
import java.io.UnsupportedEncodingException;
import java.net.URI;
import java.net.URISyntaxException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.annotation.Nullable;

import io.reactivex.annotations.Beta;
import io.realm.annotations.RealmModule;
import io.realm.exceptions.RealmException;
import io.realm.internal.OsRealmConfig;
import io.realm.internal.RealmProxyMediator;
import io.realm.internal.Util;
import io.realm.internal.sync.permissions.ObjectPermissionsModule;
import io.realm.log.RealmLog;
import io.realm.rx.RealmObservableFactory;
import io.realm.rx.RxObservableFactory;


public class SyncConfiguration extends RealmConfiguration {



    static final int MAX_FULL_PATH_LENGTH = 256;
    static final int MAX_FILE_NAME_LENGTH = 255;
    private static final char[] INVALID_CHARS = {'<', '>', ':', '"', '/', '\\', '|', '?', '*'};
    private final URI serverUrl;
    private final SyncUser user;
    private final SyncSession.ErrorHandler errorHandler;
    private final boolean deleteRealmOnLogout;
    private final boolean syncClientValidateSsl;
    @Nullable
    private final String serverCertificateAssetName;
    @Nullable
    private final String serverCertificateFilePath;
    private final boolean waitForInitialData;
    private final OsRealmConfig.SyncSessionStopPolicy sessionStopPolicy;
    private final boolean isPartial;

    private SyncConfiguration(File directory,
                                String filename,
                                String canonicalPath,
                                @Nullable
                                String assetFilePath,
                                @Nullable
                                byte[] key,
                                long schemaVersion,
                                @Nullable
                                RealmMigration migration,
                                boolean deleteRealmIfMigrationNeeded,
                                OsRealmConfig.Durability durability,
                                RealmProxyMediator schemaMediator,
                                @Nullable
                                RxObservableFactory rxFactory,
                                @Nullable
                                Realm.Transaction initialDataTransaction,
                                boolean readOnly,
                                SyncUser user,
                                URI serverUrl,
                                SyncSession.ErrorHandler errorHandler,
                                boolean deleteRealmOnLogout,
                                boolean syncClientValidateSsl,
                                @Nullable
                                String serverCertificateAssetName,
                                @Nullable
                                String serverCertificateFilePath,
                                boolean waitForInitialData,
                                OsRealmConfig.SyncSessionStopPolicy sessionStopPolicy,
                                boolean isPartial
    ) {
        super(directory,
                filename,
                canonicalPath,
                assetFilePath,
                key,
                schemaVersion,
                migration,
                deleteRealmIfMigrationNeeded,
                durability,
                schemaMediator,
                rxFactory,
                initialDataTransaction,
                readOnly,
                null,
                false
        );

        this.user = user;
        this.serverUrl = serverUrl;
        this.errorHandler = errorHandler;
        this.deleteRealmOnLogout = deleteRealmOnLogout;
        this.syncClientValidateSsl = syncClientValidateSsl;
        this.serverCertificateAssetName = serverCertificateAssetName;
        this.serverCertificateFilePath = serverCertificateFilePath;
        this.waitForInitialData = waitForInitialData;
        this.sessionStopPolicy = sessionStopPolicy;
        this.isPartial = isPartial;
    }

    
    public static RealmConfiguration forRecovery(String canonicalPath, @Nullable byte[] encryptionKey, @Nullable Object... modules) {
        HashSet<Object> validatedModules = new HashSet<>();
        if (modules != null && modules.length > 0) {
            for (Object module : modules) {
                if (!module.getClass().isAnnotationPresent(RealmModule.class)) {
                    throw new IllegalArgumentException(module.getClass().getCanonicalName() + " is not a RealmModule. " +
                            "Add @RealmModule to the class definition.");
                }
                validatedModules.add(module);
            }
        } else {
            if (Realm.getDefaultModule() != null) {
                validatedModules.add(Realm.getDefaultModule());
            }
        }

        RealmProxyMediator schemaMediator = createSchemaMediator(validatedModules, Collections.<Class<? extends RealmModel>>emptySet());
        return forRecovery(canonicalPath, encryptionKey, schemaMediator);
    }

    
    public static RealmConfiguration forRecovery(String canonicalPath) {
        return forRecovery(canonicalPath, null);
    }

    static RealmConfiguration forRecovery(String canonicalPath, @Nullable byte[] encryptionKey, RealmProxyMediator schemaMediator) {
        return new RealmConfiguration(null,null, canonicalPath,null, encryptionKey, 0,null, false, OsRealmConfig.Durability.FULL, schemaMediator, null, null, true, null, true);
    }

    static URI resolveServerUrl(URI serverUrl, String userIdentifier) {
        try {
            return new URI(serverUrl.toString().replace("/~/", "/" + userIdentifier + "/"));
        } catch (URISyntaxException e) {
            throw new IllegalArgumentException("Could not replace '/~/' with a valid user ID.", e);
        }
    }

    
    @Deprecated
    @Beta
    public static SyncConfiguration automatic() {
        SyncUser user = SyncUser.current();
        if (user == null) {
            throw new IllegalStateException("No user was logged in.");
        }
        return user.getDefaultConfiguration();
    }

    
    @Deprecated
    @Beta
    public static SyncConfiguration automatic(SyncUser user) {
        if (user == null) {
            throw new IllegalArgumentException("Non-null 'user' required.");
        }
        if (!user.isValid()) {
            throw new IllegalArgumentException("User is no logger valid.  Log the user in again.");
        }
        return user.getDefaultConfiguration();
    }


    private static String getServerPath(URI serverUrl) {
        String path = serverUrl.getPath();
        int endIndex = path.lastIndexOf("/");
        if (endIndex == -1 ) {
            return path;
        } else if (endIndex == 0) {
            return path.substring(1);
        } else {
            return path.substring(1, endIndex); 
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;

        SyncConfiguration that = (SyncConfiguration) o;

        if (deleteRealmOnLogout != that.deleteRealmOnLogout) return false;
        if (syncClientValidateSsl != that.syncClientValidateSsl) return false;
        if (!serverUrl.equals(that.serverUrl)) return false;
        if (!user.equals(that.user)) return false;
        if (!errorHandler.equals(that.errorHandler)) return false;
        if (serverCertificateAssetName != null ? !serverCertificateAssetName.equals(that.serverCertificateAssetName) : that.serverCertificateAssetName != null) return false;
        if (serverCertificateFilePath != null ? !serverCertificateFilePath.equals(that.serverCertificateFilePath) : that.serverCertificateFilePath != null) return false;
        if (waitForInitialData != that.waitForInitialData) return false;
        return true;
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + serverUrl.hashCode();
        result = 31 * result + user.hashCode();
        result = 31 * result + errorHandler.hashCode();
        result = 31 * result + (deleteRealmOnLogout ? 1 : 0);
        result = 31 * result + (syncClientValidateSsl ? 1 : 0);
        result = 31 * result + (serverCertificateAssetName != null ? serverCertificateAssetName.hashCode() : 0);
        result = 31 * result + (serverCertificateFilePath != null ? serverCertificateFilePath.hashCode() : 0);
        result = 31 * result + (waitForInitialData ? 1 : 0);
        return result;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder(super.toString());
        stringBuilder.append("\n");
        stringBuilder.append("serverUrl: " + serverUrl);
        stringBuilder.append("\n");
        stringBuilder.append("user: " + user);
        stringBuilder.append("\n");
        stringBuilder.append("errorHandler: " + errorHandler);
        stringBuilder.append("\n");
        stringBuilder.append("deleteRealmOnLogout: " + deleteRealmOnLogout);
        stringBuilder.append("\n");
        stringBuilder.append("waitForInitialRemoteData: " + waitForInitialData);
        return stringBuilder.toString();
    }

    
    public SyncUser getUser() {
        return user;
    }

    
    public URI getServerUrl() {
        return serverUrl;
    }

    public SyncSession.ErrorHandler getErrorHandler() {
        return errorHandler;
    }

    
    public boolean shouldDeleteRealmOnLogout() {
        return deleteRealmOnLogout;
    }

    
    @Nullable
    public String getServerCertificateAssetName() {
        return serverCertificateAssetName;
    }

    
    @Nullable
    public String getServerCertificateFilePath() {
        return serverCertificateFilePath;
    }

    
    public boolean syncClientValidateSsl() {
        return syncClientValidateSsl;
    }

    
    public boolean shouldWaitForInitialRemoteData() {
        return waitForInitialData;
    }

    @Override
    boolean isSyncConfiguration() {
        return true;
    }

    
    public OsRealmConfig.SyncSessionStopPolicy getSessionStopPolicy() {
        return sessionStopPolicy;
    }

    
    @Deprecated
    public boolean isPartialRealm() {
        return isPartial;
    }

    
    public boolean isFullySynchronizedRealm() {
        return !isPartial;
    }

    
    public static final class Builder  {

        private File directory;
        private boolean overrideDefaultFolder = false;
        private String fileName;
        private boolean overrideDefaultLocalFileName = false;
        @Nullable
        private byte[] key;
        private long schemaVersion = 0;
        private HashSet<Object> modules = new HashSet<Object>();
        private HashSet<Class<? extends RealmModel>> debugSchema = new HashSet<Class<? extends RealmModel>>();
        @Nullable
        private RxObservableFactory rxFactory;
        @Nullable
        private Realm.Transaction initialDataTransaction;
        private File defaultFolder;
        private String defaultLocalFileName;
        private OsRealmConfig.Durability durability = OsRealmConfig.Durability.FULL;
        private final Pattern pattern = Pattern.compile("^[A-Za-z0-9_\\-\\.]+$"); 
        private boolean readOnly = false;
        private boolean waitForServerChanges = false;

        private boolean deleteRealmOnLogout = false;
        private URI serverUrl;
        private SyncUser user = null;
        private SyncSession.ErrorHandler errorHandler = SyncManager.defaultSessionErrorHandler;
        private boolean syncClientValidateSsl = true;
        @Nullable
        private String serverCertificateAssetName;
        @Nullable
        private String serverCertificateFilePath;
        private OsRealmConfig.SyncSessionStopPolicy sessionStopPolicy = OsRealmConfig.SyncSessionStopPolicy.AFTER_CHANGES_UPLOADED;
        private boolean isPartial = true; 
        
        @Deprecated
        public Builder(SyncUser user, String uri) {
            this(BaseRealm.applicationContext, user, uri);
            fullSynchronization();
        }

        Builder(Context context, SyncUser user, String url) {

            if (context == null) {
                throw new IllegalStateException("Call `Realm.init(Context)` before creating a SyncConfiguration");
            }
            this.defaultFolder = new File(context.getFilesDir(), "realm-object-server");
            if (Realm.getDefaultModule() != null) {
                this.modules.add(Realm.getDefaultModule());
            }

            validateAndSet(user);
            validateAndSet(url);
        }

        private void validateAndSet(SyncUser user) {

            if (user == null) {
                throw new IllegalArgumentException("Non-null `user` required.");
            }
            if (!user.isValid()) {
                throw new IllegalArgumentException("User not authenticated or authentication expired.");
            }
            this.user = user;
        }

        private void validateAndSet(String uri) {

            if (uri == null) {
                throw new IllegalArgumentException("Non-null 'uri' required.");
            }

            try {
                serverUrl = new URI(uri);
            } catch (URISyntaxException e) {
                throw new IllegalArgumentException("Invalid URI: " + uri, e);
            }

            try {

                String serverScheme = serverUrl.getScheme();
                if (serverScheme == null) {
                    String authProtocol = user.getAuthenticationUrl().getProtocol();
                    if (authProtocol.equalsIgnoreCase("https")) {
                        serverScheme = "realms";
                    } else {
                        serverScheme = "realm";
                    }
                } else if (serverScheme.equalsIgnoreCase("http")) {
                    serverScheme = "realm";
                } else if (serverScheme.equalsIgnoreCase("https")) {
                    serverScheme = "realms";
                }


                String host = serverUrl.getHost();
                if (host == null) {
                    host = user.getAuthenticationUrl().getHost();
                }


                String path = serverUrl.getPath();
                if (path != null && !path.startsWith("/")) {
                    path = "/" + path;
                }

                serverUrl = new URI(serverScheme,
                        serverUrl.getUserInfo(),
                        host,
                        serverUrl.getPort(),
                        (path != null) ? path.replace(host + "/", "") : null, 
                        serverUrl.getQuery(),
                        serverUrl.getRawFragment());

            } catch (URISyntaxException e) {
                throw new IllegalArgumentException("Invalid URI: " + uri, e);
            }


            String path = serverUrl.getPath();
            if (path == null) {
                throw new IllegalArgumentException("Invalid URI: " + uri);
            }

            String[] pathSegments = path.split("/");
            for (int i = 1; i < pathSegments.length; i++) {
                String segment = pathSegments[i];
                if (segment.equals("~")) {
                    continue;
                }
                if (segment.equals("..") || segment.equals(".")) {
                    throw new IllegalArgumentException("The URI has an invalid segment: " + segment);
                }
                Matcher m = pattern.matcher(segment);
                if (!m.matches()) {
                    throw new IllegalArgumentException("The URI must only contain characters 0-9, a-z, A-Z, ., _, and -: " + segment);
                }
            }

            this.defaultLocalFileName = pathSegments[pathSegments.length - 1];


            
            if (defaultLocalFileName.endsWith(".realm")
                    || defaultLocalFileName.endsWith(".realm.lock")
                    || defaultLocalFileName.endsWith(".realm.management")) {
                throw new IllegalArgumentException("The URI must not end with '.realm', '.realm.lock' or '.realm.management: " + uri);
            }
        }

        
        public Builder name(String filename) {

            if (filename == null || filename.isEmpty()) {
                throw new IllegalArgumentException("A non-empty filename must be provided");
            }
            this.fileName = filename;
            this.overrideDefaultLocalFileName = true;
            return this;
        }

        
        public Builder directory(File directory) {

            if (directory == null) {
                throw new IllegalArgumentException("Non-null 'directory' required.");
            }
            if (directory.isFile()) {
                throw new IllegalArgumentException("'directory' is a file, not a directory: " +
                        directory.getAbsolutePath() + ".");
            }
            if (!directory.exists() && !directory.mkdirs()) {
                throw new IllegalArgumentException("Could not create the specified directory: " +
                        directory.getAbsolutePath() + ".");
            }
            if (!directory.canWrite()) {
                throw new IllegalArgumentException("Realm directory is not writable: " +
                        directory.getAbsolutePath() + ".");
            }
            this.directory = directory;
            overrideDefaultFolder = true;
            return this;
        }

        
        public Builder encryptionKey(byte[] key) {

            if (key == null) {
                throw new IllegalArgumentException("A non-null key must be provided");
            }
            if (key.length != KEY_LENGTH) {
                throw new IllegalArgumentException(String.format(Locale.US,
                        "The provided key must be %s bytes. Yours was: %s",
                        KEY_LENGTH, key.length));
            }
            this.key = Arrays.copyOf(key, key.length);
            return this;
        }

        
        SyncConfiguration.Builder schema(Class<? extends RealmModel> firstClass, Class<? extends RealmModel>... additionalClasses) {

            if (firstClass == null) {
                throw new IllegalArgumentException("A non-null class must be provided");
            }
            modules.clear();
            modules.add(DEFAULT_MODULE_MEDIATOR);
            debugSchema.add(firstClass);

            if (additionalClasses != null) {
                Collections.addAll(debugSchema, additionalClasses);
            }

            return this;
        }

        
        SyncConfiguration.Builder sessionStopPolicy(OsRealmConfig.SyncSessionStopPolicy policy) {
            sessionStopPolicy = policy;
            return this;
        }


        
        public Builder schemaVersion(long schemaVersion) {
            if (schemaVersion < 0) {
                throw new IllegalArgumentException("Realm schema version numbers must be 0 (zero) or higher. Yours was: " + schemaVersion);
            }
            this.schemaVersion = schemaVersion;
            return this;
        }

        
        public Builder modules(Object baseModule, Object... additionalModules) {
            modules.clear();
            addModule(baseModule);

            if (additionalModules != null) {
                for (Object module : additionalModules) {
                    addModule(module);
                }
            }
            return this;
        }

        
        public Builder modules(Iterable<Object> modules) {
            this.modules.clear();
            if (modules != null) {
                for (Object module : modules) {
                    addModule(module);
                }
            }
            return this;
        }

        
        public Builder addModule(Object module) {

            if (module != null) {
                checkModule(module);
                modules.add(module);
            }

            return this;
        }

        
        public Builder rxFactory(RxObservableFactory factory) {
            rxFactory = factory;
            return this;
        }

        
        public Builder initialData(Realm.Transaction transaction) {
            initialDataTransaction = transaction;
            return this;
        }

        
        public Builder inMemory() {
            this.durability = OsRealmConfig.Durability.MEM_ONLY;
            return this;
        }

        
        public Builder errorHandler(SyncSession.ErrorHandler errorHandler) {

            if (errorHandler == null) {
                throw new IllegalArgumentException("Non-null 'errorHandler' required.");
            }
            this.errorHandler = errorHandler;
            return this;
        }

        
        public Builder trustedRootCA(String filename) {

            if (filename == null || filename.isEmpty()) {
                throw new IllegalArgumentException("A non-empty filename must be provided");
            }
            this.serverCertificateAssetName = filename;
            return this;
        }

        
        public Builder disableSSLVerification() {
            this.syncClientValidateSsl = false;
            return this;
        }

        
        public Builder waitForInitialRemoteData() {
            this.waitForServerChanges = true;
            return this;
        }

        
        public SyncConfiguration.Builder readOnly() {
            this.readOnly = true;
            return this;
        }

        
        @Deprecated
            return this;
        }

        
        public SyncConfiguration.Builder fullSynchronization() {
            this.isPartial = false;
            return this;
        }

        private String MD5(String in) {
            try {
                MessageDigest digest = MessageDigest.getInstance("MD5");
                byte[] buf = digest.digest(in.getBytes("UTF-8"));
                StringBuilder builder = new StringBuilder();
                for (byte b : buf) {
                    builder.append(String.format(Locale.US, "%02X", b));
                }
                return builder.toString();
            } catch (NoSuchAlgorithmException e) {
                throw new RealmException(e.getMessage());
            } catch (UnsupportedEncodingException e) {
                throw new RealmException(e.getMessage());
            }
        }

        
        

        
        public SyncConfiguration build() {
            if (serverUrl == null || user == null) {
                throw new IllegalStateException("serverUrl() and user() are both required.");
            }



            if (readOnly) {
                if (initialDataTransaction != null) {
                    throw new IllegalStateException("This Realm is marked as read-only. " +
                            "Read-only Realms cannot use initialData(Realm.Transaction).");
                }
                if (!waitForServerChanges) {
                    throw new IllegalStateException("A read-only Realms must be provided by some source. " +
                            "'waitForInitialRemoteData()' wasn't enabled which is currently the only supported source.");
                }
            }


            if (serverUrl.toString().contains("/~/") && user.getIdentity() == null) {
                throw new IllegalStateException("The serverUrl contains a /~/, but the user does not have an identity." +
                        " Most likely it hasn't been authenticated yet or has been created directly from an" +
                        " access token. Use a path without /~/.");
            }

            if (rxFactory == null && isRxJavaAvailable()) {
                rxFactory = new RealmObservableFactory();
            }




            URI resolvedServerUrl = resolveServerUrl(serverUrl, user.getIdentity());
            File rootDir = overrideDefaultFolder ? directory : defaultFolder;
            String realmPathFromRootDir = user.getIdentity() + "/" + getServerPath(resolvedServerUrl);
            File realmFileDirectory = new File(rootDir, realmPathFromRootDir);

            String realmFileName = overrideDefaultLocalFileName ? fileName : defaultLocalFileName;
            String fullPathName = realmFileDirectory.getAbsolutePath() + File.pathSeparator + realmFileName;

            if (fullPathName.length() > MAX_FULL_PATH_LENGTH) {

                realmFileName = MD5(realmFileName);
                fullPathName = realmFileDirectory.getAbsolutePath() + File.pathSeparator + realmFileName;
                if (fullPathName.length() > MAX_FULL_PATH_LENGTH) {

                    realmFileDirectory = new File(rootDir, user.getIdentity());
                    fullPathName = realmFileDirectory.getAbsolutePath() + File.pathSeparator + realmFileName;
                    if (fullPathName.length() > MAX_FULL_PATH_LENGTH) { 
                        throw new IllegalStateException(String.format(Locale.US,
                                "Full path name must not exceed %d characters: %s",
                                MAX_FULL_PATH_LENGTH, fullPathName));
                    }
                }
            }

            if (realmFileName.length() > MAX_FILE_NAME_LENGTH) {
                throw new IllegalStateException(String.format(Locale.US,
                        "File name exceed %d characters: %d", MAX_FILE_NAME_LENGTH,
                        realmFileName.length()));
            }


            for (char c : INVALID_CHARS) {
                realmFileName = realmFileName.replace(c, '_');
            }


            if (!realmFileDirectory.exists() && !realmFileDirectory.mkdirs()) {
                throw new IllegalStateException("Could not create directory for saving the Realm: " + realmFileDirectory);
            }

            if (!Util.isEmptyString(serverCertificateAssetName)) {
                if (syncClientValidateSsl) {



                    String fileName = serverCertificateAssetName.substring(serverCertificateAssetName.lastIndexOf(File.separatorChar) + 1);
                    serverCertificateFilePath = new File(realmFileDirectory, fileName).getAbsolutePath();
                } else {
                    RealmLog.warn("SSL Verification is disabled, the provided server certificate will not be used.");
                }
            }


            if (isPartial) {
                addModule(new ObjectPermissionsModule());
            }

            return new SyncConfiguration(

                    realmFileDirectory,
                    realmFileName,
                    getCanonicalPath(new File(realmFileDirectory, realmFileName)),
                    null, 
                    key,
                    schemaVersion,
                    null, 
                    false, 
                    durability,
                    createSchemaMediator(modules, debugSchema),
                    rxFactory,
                    initialDataTransaction,
                    readOnly,


                    user,
                    resolvedServerUrl,
                    errorHandler,
                    deleteRealmOnLogout,
                    syncClientValidateSsl,
                    serverCertificateAssetName,
                    serverCertificateFilePath,
                    waitForServerChanges,
                    sessionStopPolicy,
                    isPartial
            );
        }

        private void checkModule(Object module) {
            if (!module.getClass().isAnnotationPresent(RealmModule.class)) {
                throw new IllegalArgumentException(module.getClass().getCanonicalName() + " is not a RealmModule. " +
                        "Add @RealmModule to the class definition.");
            }
        }
    }
}

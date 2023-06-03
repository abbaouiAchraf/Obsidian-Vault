# EXPORT Certif and Signature(StoreKey)
![[Pasted image 20230523161825.png]]
## Keystore 
Cela vous permet de générer une paire de clés pour votre certificat SSL.

```
keytool -genkey -alias cmp -keypass cmpkey -keyalg RSA -keystore cmpstore -storepass cmpstorekey
```
-   `keytool`: This is the command to invoke the keytool utility.
    
-   `-genkey`: This option instructs keytool to generate a new key pair.
    
-   `-alias cmp`: This specifies the alias for the generated key pair. The alias is a unique identifier used to reference the key pair later.
    
-   `-keypass cmpkey`: This sets the password for the generated private key. In this case, the password is set to "cmpkey".
    
-   `-keyalg RSA`: This specifies the algorithm to be used for generating the key pair. In this case, RSA (Rivest-Shamir-Adleman) algorithm is used.
    
-   `-keystore cmpstore`: This sets the filename for the keystore where the generated key pair will be stored. The keystore is a file that holds the keys and certificates. In this case, the filename is set to "cmpstore".
    
-   `-storepass cmpstorekey`: This sets the password for the keystore. The password is used to protect the integrity of the keystore file itself. In this case, the password is set to "cmpstorekey".

When you execute this command, the keytool utility will generate a new key pair using the RSA algorithm and store it in a keystore file named "*cmpstore*". The private key will be protected by the password "*cmpkey*", and the keystore file will be protected by the password "*cmpstorekey*". The alias "*cmp*" can be used later to refer to this key pair when performing operations such as signing or encrypting data.

![[Pasted image 20230523155737.png]]

## JarSigner
used to sign Java Archive (JAR) files with digital signatures.
```
jarsigner -keystore cmpstore -signedjar secureSecurity.jar Security.jar cmp
```

Here's the explanation of each parameter:

-   `jarsigner`: This is the command to invoke the jarsigner utility.
    
-   `-keystore cmpstore`: This specifies the keystore file (`cmpstore`) that contains the private key used for signing the JAR file. The keystore should have been created previously using the `keytool` command.
    
-   `-signedjar secureSecurity.jar`: This sets the output file name (`secureSecurity.jar`) for the signed JAR file. The signed JAR file is the original `Security.jar` file with added digital signatures.
    
-   `Security.jar`: This is the original JAR file that you want to sign.
    
-   `cmp`: This is the alias (`cmp`) that represents the entry in the keystore containing the private key used for signing.

![[Pasted image 20230523161647.png]]

## Certification generation

```
Keytool -export -keystore cmpstore -alias cmp -file cerfcmp.cer
```
Here's the explanation of each parameter:

-   `Keytool`: This is the command to invoke the keytool utility. Please note that it should be lowercase as `keytool`.
    
-   `-export`: This option instructs keytool to perform an export operation.
    
-   `-keystore cmpstore`: This specifies the keystore file (`cmpstore`) from which the certificate should be exported.
    
-   `-alias cmp`: This specifies the alias (`cmp`) of the entry in the keystore for which the certificate should be exported.
    
-   `-file cerfcmp.cer`: This sets the output file name (`cerfcmp.cer`) for the exported certificate. The certificate will be saved as a `.cer` file.

![[Pasted image 20230523161709.png]]

# IMPORT Signature(StoreKey) and Certif
![[Pasted image 20230523161940.png]]
## Importation de certificat

Importing command :
```
keytool -import -alias rcmp -file cerfcmp.cer -keystore Rcmpstore
```
Here's the explanation of each parameter:

-   `keytool`: This is the command to invoke the keytool utility.
    
-   `-import`: This option instructs keytool to perform an import operation.
    
-   `-alias rcmp`: This specifies the alias (`rcmp`) for the imported certificate. The alias is a unique identifier used to reference the certificate later.
    
-   `-file cerfcmp.cer`: This specifies the input file (`cerfcmp.cer`) that contains the certificate to be imported.
    
-   `-keystore Rcmpstore`: This sets the filename (`Rcmpstore`) for the keystore where the certificate should be imported. The **keystore** is a file that holds the keys and certificates.
![[Pasted image 20230523162625.png]]
## File Policy (ACL)"Access Contol List"

Create a new ACL (Access Control List) and modify the policy file to grant access to desired resources.


## Exchange Signed Docs
Received must receive the .jar and the certification
```
keytool -import -alias recpContrat -file .\cerfcmp.cer -keystore recpstore
```
- `keytool`: This is the command to invoke the `keytool` utility.
- `-import`: This option specifies that you want to import a certificate into a keystore.
- `-alias recpContrat`: Here, `recpContrat` is the alias or identifier you are assigning to the imported certificate. This alias will be used to refer to the certificate within the keystore.
- `-file .\cerfcmp.cer`: This option indicates the path to the certificate file (`cerfcmp.cer` in the current directory, denoted by `.\`). The `keytool` utility will read this certificate file for import.
- `-keystore recpstore`: This option specifies the filename of the keystore where the certificate will be stored or updated. In this case, the keystore file is named `recpstore`.

## Verify the signature and the .jar

```
jarsigner -verify -verbose -keystore recpstore sContrat.jar
```
- `jarsigner`: This is the command to invoke the `jarsigner` utility.
- `-verify`: This option instructs `jarsigner` to verify the digital signature of the JAR file.
- `-verbose`: This option enables verbose output, providing additional information during the verification process.
- `-keystore recpstore`: Here, `recpstore` refers to the keystore file that contains the necessary keys and certificates required for the verification process.
- `sContrat.jar`: This specifies the JAR file (`sContrat.jar`) that you want to verify.

In summary, the command is using the `jarsigner` utility to verify the digital signature of the `sContrat.jar` JAR file using the keys and certificates stored in the `recpstore` keystore. The `-verbose` option is used to obtain more detailed information about the verification process.


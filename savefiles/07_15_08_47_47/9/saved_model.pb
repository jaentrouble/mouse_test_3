же
еЃ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*	2.3.0-rc12v2.3.0-rc0-15-g14b2d686d68
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Р*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
Р*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:@*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:@*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
t
cond_1/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *!
shared_namecond_1/Adam/iter
m
$cond_1/Adam/iter/Read/ReadVariableOpReadVariableOpcond_1/Adam/iter*
_output_shapes
: *
dtype0	
x
cond_1/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namecond_1/Adam/beta_1
q
&cond_1/Adam/beta_1/Read/ReadVariableOpReadVariableOpcond_1/Adam/beta_1*
_output_shapes
: *
dtype0
x
cond_1/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namecond_1/Adam/beta_2
q
&cond_1/Adam/beta_2/Read/ReadVariableOpReadVariableOpcond_1/Adam/beta_2*
_output_shapes
: *
dtype0
v
cond_1/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namecond_1/Adam/decay
o
%cond_1/Adam/decay/Read/ReadVariableOpReadVariableOpcond_1/Adam/decay*
_output_shapes
: *
dtype0

cond_1/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namecond_1/Adam/learning_rate

-cond_1/Adam/learning_rate/Read/ReadVariableOpReadVariableOpcond_1/Adam/learning_rate*
_output_shapes
: *
dtype0
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:	@*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:@*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:@ *
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
: *
dtype0
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
: *
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
:*
dtype0
~
conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_nameconv1d_3/kernel
w
#conv1d_3/kernel/Read/ReadVariableOpReadVariableOpconv1d_3/kernel*"
_output_shapes
:	@*
dtype0
r
conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_3/bias
k
!conv1d_3/bias/Read/ReadVariableOpReadVariableOpconv1d_3/bias*
_output_shapes
:@*
dtype0
~
conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv1d_4/kernel
w
#conv1d_4/kernel/Read/ReadVariableOpReadVariableOpconv1d_4/kernel*"
_output_shapes
:@ *
dtype0
r
conv1d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_4/bias
k
!conv1d_4/bias/Read/ReadVariableOpReadVariableOpconv1d_4/bias*
_output_shapes
: *
dtype0
~
conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv1d_5/kernel
w
#conv1d_5/kernel/Read/ReadVariableOpReadVariableOpconv1d_5/kernel*"
_output_shapes
: *
dtype0
r
conv1d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d_5/bias
k
!conv1d_5/bias/Read/ReadVariableOpReadVariableOpconv1d_5/bias*
_output_shapes
:*
dtype0
x
current_loss_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namecurrent_loss_scale
q
&current_loss_scale/Read/ReadVariableOpReadVariableOpcurrent_loss_scale*
_output_shapes
: *
dtype0
h

good_stepsVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
good_steps
a
good_steps/Read/ReadVariableOpReadVariableOp
good_steps*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

cond_1/Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Р*+
shared_namecond_1/Adam/dense/kernel/m

.cond_1/Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense/kernel/m* 
_output_shapes
:
Р*
dtype0

cond_1/Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namecond_1/Adam/dense/bias/m

,cond_1/Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense/bias/m*
_output_shapes	
:*
dtype0

cond_1/Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_namecond_1/Adam/dense_1/kernel/m

0cond_1/Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense_1/kernel/m* 
_output_shapes
:
*
dtype0

cond_1/Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namecond_1/Adam/dense_1/bias/m

.cond_1/Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense_1/bias/m*
_output_shapes	
:*
dtype0

cond_1/Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*-
shared_namecond_1/Adam/dense_2/kernel/m

0cond_1/Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense_2/kernel/m*
_output_shapes
:	@*
dtype0

cond_1/Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namecond_1/Adam/dense_2/bias/m

.cond_1/Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense_2/bias/m*
_output_shapes
:@*
dtype0

cond_1/Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*-
shared_namecond_1/Adam/dense_3/kernel/m

0cond_1/Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense_3/kernel/m*
_output_shapes

:@*
dtype0

cond_1/Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namecond_1/Adam/dense_3/bias/m

.cond_1/Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense_3/bias/m*
_output_shapes
:*
dtype0

cond_1/Adam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*,
shared_namecond_1/Adam/conv1d/kernel/m

/cond_1/Adam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d/kernel/m*"
_output_shapes
:	@*
dtype0

cond_1/Adam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namecond_1/Adam/conv1d/bias/m

-cond_1/Adam/conv1d/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d/bias/m*
_output_shapes
:@*
dtype0

cond_1/Adam/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *.
shared_namecond_1/Adam/conv1d_1/kernel/m

1cond_1/Adam/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_1/kernel/m*"
_output_shapes
:@ *
dtype0

cond_1/Adam/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namecond_1/Adam/conv1d_1/bias/m

/cond_1/Adam/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_1/bias/m*
_output_shapes
: *
dtype0

cond_1/Adam/conv1d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namecond_1/Adam/conv1d_2/kernel/m

1cond_1/Adam/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_2/kernel/m*"
_output_shapes
: *
dtype0

cond_1/Adam/conv1d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/conv1d_2/bias/m

/cond_1/Adam/conv1d_2/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_2/bias/m*
_output_shapes
:*
dtype0

cond_1/Adam/conv1d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*.
shared_namecond_1/Adam/conv1d_3/kernel/m

1cond_1/Adam/conv1d_3/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_3/kernel/m*"
_output_shapes
:	@*
dtype0

cond_1/Adam/conv1d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namecond_1/Adam/conv1d_3/bias/m

/cond_1/Adam/conv1d_3/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_3/bias/m*
_output_shapes
:@*
dtype0

cond_1/Adam/conv1d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *.
shared_namecond_1/Adam/conv1d_4/kernel/m

1cond_1/Adam/conv1d_4/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_4/kernel/m*"
_output_shapes
:@ *
dtype0

cond_1/Adam/conv1d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namecond_1/Adam/conv1d_4/bias/m

/cond_1/Adam/conv1d_4/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_4/bias/m*
_output_shapes
: *
dtype0

cond_1/Adam/conv1d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namecond_1/Adam/conv1d_5/kernel/m

1cond_1/Adam/conv1d_5/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_5/kernel/m*"
_output_shapes
: *
dtype0

cond_1/Adam/conv1d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/conv1d_5/bias/m

/cond_1/Adam/conv1d_5/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_5/bias/m*
_output_shapes
:*
dtype0

cond_1/Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Р*+
shared_namecond_1/Adam/dense/kernel/v

.cond_1/Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense/kernel/v* 
_output_shapes
:
Р*
dtype0

cond_1/Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namecond_1/Adam/dense/bias/v

,cond_1/Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense/bias/v*
_output_shapes	
:*
dtype0

cond_1/Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_namecond_1/Adam/dense_1/kernel/v

0cond_1/Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense_1/kernel/v* 
_output_shapes
:
*
dtype0

cond_1/Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namecond_1/Adam/dense_1/bias/v

.cond_1/Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense_1/bias/v*
_output_shapes	
:*
dtype0

cond_1/Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*-
shared_namecond_1/Adam/dense_2/kernel/v

0cond_1/Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense_2/kernel/v*
_output_shapes
:	@*
dtype0

cond_1/Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namecond_1/Adam/dense_2/bias/v

.cond_1/Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense_2/bias/v*
_output_shapes
:@*
dtype0

cond_1/Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*-
shared_namecond_1/Adam/dense_3/kernel/v

0cond_1/Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense_3/kernel/v*
_output_shapes

:@*
dtype0

cond_1/Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namecond_1/Adam/dense_3/bias/v

.cond_1/Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/dense_3/bias/v*
_output_shapes
:*
dtype0

cond_1/Adam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*,
shared_namecond_1/Adam/conv1d/kernel/v

/cond_1/Adam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d/kernel/v*"
_output_shapes
:	@*
dtype0

cond_1/Adam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namecond_1/Adam/conv1d/bias/v

-cond_1/Adam/conv1d/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d/bias/v*
_output_shapes
:@*
dtype0

cond_1/Adam/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *.
shared_namecond_1/Adam/conv1d_1/kernel/v

1cond_1/Adam/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_1/kernel/v*"
_output_shapes
:@ *
dtype0

cond_1/Adam/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namecond_1/Adam/conv1d_1/bias/v

/cond_1/Adam/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_1/bias/v*
_output_shapes
: *
dtype0

cond_1/Adam/conv1d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namecond_1/Adam/conv1d_2/kernel/v

1cond_1/Adam/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_2/kernel/v*"
_output_shapes
: *
dtype0

cond_1/Adam/conv1d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/conv1d_2/bias/v

/cond_1/Adam/conv1d_2/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_2/bias/v*
_output_shapes
:*
dtype0

cond_1/Adam/conv1d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*.
shared_namecond_1/Adam/conv1d_3/kernel/v

1cond_1/Adam/conv1d_3/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_3/kernel/v*"
_output_shapes
:	@*
dtype0

cond_1/Adam/conv1d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namecond_1/Adam/conv1d_3/bias/v

/cond_1/Adam/conv1d_3/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_3/bias/v*
_output_shapes
:@*
dtype0

cond_1/Adam/conv1d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *.
shared_namecond_1/Adam/conv1d_4/kernel/v

1cond_1/Adam/conv1d_4/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_4/kernel/v*"
_output_shapes
:@ *
dtype0

cond_1/Adam/conv1d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namecond_1/Adam/conv1d_4/bias/v

/cond_1/Adam/conv1d_4/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_4/bias/v*
_output_shapes
: *
dtype0

cond_1/Adam/conv1d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namecond_1/Adam/conv1d_5/kernel/v

1cond_1/Adam/conv1d_5/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_5/kernel/v*"
_output_shapes
: *
dtype0

cond_1/Adam/conv1d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/conv1d_5/bias/v

/cond_1/Adam/conv1d_5/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1d_5/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
П}
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*њ|
value№|Bэ| Bц|
Т
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
#_self_saveable_object_factories
	optimizer

signatures
regularization_losses
	variables
trainable_variables
	keras_api
%
#_self_saveable_object_factories
%
#_self_saveable_object_factories

layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api

layer-0
 layer-1
!layer_with_weights-0
!layer-2
"layer_with_weights-1
"layer-3
#layer_with_weights-2
#layer-4
#$_self_saveable_object_factories
%regularization_losses
&	variables
'trainable_variables
(	keras_api
w
#)_self_saveable_object_factories
*regularization_losses
+	variables
,trainable_variables
-	keras_api
w
#._self_saveable_object_factories
/regularization_losses
0	variables
1trainable_variables
2	keras_api


3kernel
4bias
#5_self_saveable_object_factories
6regularization_losses
7	variables
8trainable_variables
9	keras_api


:kernel
;bias
#<_self_saveable_object_factories
=regularization_losses
>	variables
?trainable_variables
@	keras_api


Akernel
Bbias
#C_self_saveable_object_factories
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api


Hkernel
Ibias
#J_self_saveable_object_factories
Kregularization_losses
L	variables
Mtrainable_variables
N	keras_api
w
#O_self_saveable_object_factories
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
 
є
T
loss_scale
Ubase_optimizer
Viter

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rate3mј4mљ:mњ;mћAmќBm§HmўImџ[m\m]m^m_m`mambmcmdmemfm3v4v:v;vAvBvHvIv[v\v]v^v_v`vavbvcvdvevfv
 
 

[0
\1
]2
^3
_4
`5
a6
b7
c8
d9
e10
f11
312
413
:14
;15
A16
B17
H18
I19

[0
\1
]2
^3
_4
`5
a6
b7
c8
d9
e10
f11
312
413
:14
;15
A16
B17
H18
I19
­
regularization_losses
gmetrics
	variables
hnon_trainable_variables
trainable_variables
ilayer_metrics

jlayers
klayer_regularization_losses
 
 
%
#l_self_saveable_object_factories
w
#m_self_saveable_object_factories
nregularization_losses
o	variables
ptrainable_variables
q	keras_api


[kernel
\bias
#r_self_saveable_object_factories
sregularization_losses
t	variables
utrainable_variables
v	keras_api


]kernel
^bias
#w_self_saveable_object_factories
xregularization_losses
y	variables
ztrainable_variables
{	keras_api


_kernel
`bias
#|_self_saveable_object_factories
}regularization_losses
~	variables
trainable_variables
	keras_api
 
 
*
[0
\1
]2
^3
_4
`5
*
[0
\1
]2
^3
_4
`5
В
regularization_losses
metrics
	variables
non_trainable_variables
trainable_variables
layer_metrics
layers
 layer_regularization_losses
&
$_self_saveable_object_factories
|
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api


akernel
bbias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api


ckernel
dbias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api


ekernel
fbias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
 
 
*
a0
b1
c2
d3
e4
f5
*
a0
b1
c2
d3
e4
f5
В
%regularization_losses
metrics
&	variables
non_trainable_variables
'trainable_variables
layer_metrics
layers
 layer_regularization_losses
 
 
 
 
В
*regularization_losses
 metrics
+	variables
Ёnon_trainable_variables
,trainable_variables
Ђlayer_metrics
Ѓlayers
 Єlayer_regularization_losses
 
 
 
 
В
/regularization_losses
Ѕmetrics
0	variables
Іnon_trainable_variables
1trainable_variables
Їlayer_metrics
Јlayers
 Љlayer_regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

30
41

30
41
В
6regularization_losses
Њmetrics
7	variables
Ћnon_trainable_variables
8trainable_variables
Ќlayer_metrics
­layers
 Ўlayer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

:0
;1

:0
;1
В
=regularization_losses
Џmetrics
>	variables
Аnon_trainable_variables
?trainable_variables
Бlayer_metrics
Вlayers
 Гlayer_regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

A0
B1

A0
B1
В
Dregularization_losses
Дmetrics
E	variables
Еnon_trainable_variables
Ftrainable_variables
Жlayer_metrics
Зlayers
 Иlayer_regularization_losses
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

H0
I1

H0
I1
В
Kregularization_losses
Йmetrics
L	variables
Кnon_trainable_variables
Mtrainable_variables
Лlayer_metrics
Мlayers
 Нlayer_regularization_losses
 
 
 
 
В
Pregularization_losses
Оmetrics
Q	variables
Пnon_trainable_variables
Rtrainable_variables
Рlayer_metrics
Сlayers
 Тlayer_regularization_losses
 
 
OM
VARIABLE_VALUEcond_1/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcond_1/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcond_1/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcond_1/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEcond_1/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv1d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv1d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv1d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv1d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv1d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv1d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv1d_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv1d_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv1d_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv1d_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv1d_5/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv1d_5/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE

Х0
Ц1
 
 
N
0
1
2
3
4
5
6
7
	8

9
10
 
 
 
 
 
 
В
nregularization_losses
Чmetrics
o	variables
Шnon_trainable_variables
ptrainable_variables
Щlayer_metrics
Ъlayers
 Ыlayer_regularization_losses
 
 

[0
\1

[0
\1
В
sregularization_losses
Ьmetrics
t	variables
Эnon_trainable_variables
utrainable_variables
Юlayer_metrics
Яlayers
 аlayer_regularization_losses
 
 

]0
^1

]0
^1
В
xregularization_losses
бmetrics
y	variables
вnon_trainable_variables
ztrainable_variables
гlayer_metrics
дlayers
 еlayer_regularization_losses
 
 

_0
`1

_0
`1
В
}regularization_losses
жmetrics
~	variables
зnon_trainable_variables
trainable_variables
иlayer_metrics
йlayers
 кlayer_regularization_losses
 
 
 
#
0
1
2
3
4
 
 
 
 
 
 
Е
regularization_losses
лmetrics
	variables
мnon_trainable_variables
trainable_variables
нlayer_metrics
оlayers
 пlayer_regularization_losses
 
 

a0
b1

a0
b1
Е
regularization_losses
рmetrics
	variables
сnon_trainable_variables
trainable_variables
тlayer_metrics
уlayers
 фlayer_regularization_losses
 
 

c0
d1

c0
d1
Е
regularization_losses
хmetrics
	variables
цnon_trainable_variables
trainable_variables
чlayer_metrics
шlayers
 щlayer_regularization_losses
 
 

e0
f1

e0
f1
Е
regularization_losses
ъmetrics
	variables
ыnon_trainable_variables
trainable_variables
ьlayer_metrics
эlayers
 юlayer_regularization_losses
 
 
 
#
0
 1
!2
"3
#4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
jh
VARIABLE_VALUEcurrent_loss_scaleBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUE
good_steps:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUE
8

яtotal

№count
ё	variables
ђ	keras_api
I

ѓtotal

єcount
ѕ
_fn_kwargs
і	variables
ї	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

я0
№1

ё	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

ѓ0
є1

і	variables

VARIABLE_VALUEcond_1/Adam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEcond_1/Adam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEcond_1/Adam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/dense_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEcond_1/Adam/dense_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/dense_3/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEcond_1/Adam/dense_3/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEcond_1/Adam/conv1d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEcond_1/Adam/conv1d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEcond_1/Adam/conv1d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEcond_1/Adam/conv1d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEcond_1/Adam/conv1d_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEcond_1/Adam/conv1d_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEcond_1/Adam/conv1d_3/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEcond_1/Adam/conv1d_3/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEcond_1/Adam/conv1d_4/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEcond_1/Adam/conv1d_4/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEcond_1/Adam/conv1d_5/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEcond_1/Adam/conv1d_5/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEcond_1/Adam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEcond_1/Adam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/dense_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEcond_1/Adam/dense_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/dense_3/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEcond_1/Adam/dense_3/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEcond_1/Adam/conv1d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEcond_1/Adam/conv1d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEcond_1/Adam/conv1d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEcond_1/Adam/conv1d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEcond_1/Adam/conv1d_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEcond_1/Adam/conv1d_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEcond_1/Adam/conv1d_3/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEcond_1/Adam/conv1d_3/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEcond_1/Adam/conv1d_4/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEcond_1/Adam/conv1d_4/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEcond_1/Adam/conv1d_5/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEcond_1/Adam/conv1d_5/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_LeftPlaceholder*/
_output_shapes
:џџџџџџџџџd*
dtype0*$
shape:џџџџџџџџџd

serving_default_RightPlaceholder*/
_output_shapes
:џџџџџџџџџd*
dtype0*$
shape:џџџџџџџџџd
І
StatefulPartitionedCallStatefulPartitionedCallserving_default_Leftserving_default_Rightconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasconv1d_4/kernelconv1d_4/biasconv1d_5/kernelconv1d_5/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 */
f*R(
&__inference_signature_wrapper_33197310
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
н
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp$cond_1/Adam/iter/Read/ReadVariableOp&cond_1/Adam/beta_1/Read/ReadVariableOp&cond_1/Adam/beta_2/Read/ReadVariableOp%cond_1/Adam/decay/Read/ReadVariableOp-cond_1/Adam/learning_rate/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp#conv1d_3/kernel/Read/ReadVariableOp!conv1d_3/bias/Read/ReadVariableOp#conv1d_4/kernel/Read/ReadVariableOp!conv1d_4/bias/Read/ReadVariableOp#conv1d_5/kernel/Read/ReadVariableOp!conv1d_5/bias/Read/ReadVariableOp&current_loss_scale/Read/ReadVariableOpgood_steps/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.cond_1/Adam/dense/kernel/m/Read/ReadVariableOp,cond_1/Adam/dense/bias/m/Read/ReadVariableOp0cond_1/Adam/dense_1/kernel/m/Read/ReadVariableOp.cond_1/Adam/dense_1/bias/m/Read/ReadVariableOp0cond_1/Adam/dense_2/kernel/m/Read/ReadVariableOp.cond_1/Adam/dense_2/bias/m/Read/ReadVariableOp0cond_1/Adam/dense_3/kernel/m/Read/ReadVariableOp.cond_1/Adam/dense_3/bias/m/Read/ReadVariableOp/cond_1/Adam/conv1d/kernel/m/Read/ReadVariableOp-cond_1/Adam/conv1d/bias/m/Read/ReadVariableOp1cond_1/Adam/conv1d_1/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv1d_1/bias/m/Read/ReadVariableOp1cond_1/Adam/conv1d_2/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv1d_2/bias/m/Read/ReadVariableOp1cond_1/Adam/conv1d_3/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv1d_3/bias/m/Read/ReadVariableOp1cond_1/Adam/conv1d_4/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv1d_4/bias/m/Read/ReadVariableOp1cond_1/Adam/conv1d_5/kernel/m/Read/ReadVariableOp/cond_1/Adam/conv1d_5/bias/m/Read/ReadVariableOp.cond_1/Adam/dense/kernel/v/Read/ReadVariableOp,cond_1/Adam/dense/bias/v/Read/ReadVariableOp0cond_1/Adam/dense_1/kernel/v/Read/ReadVariableOp.cond_1/Adam/dense_1/bias/v/Read/ReadVariableOp0cond_1/Adam/dense_2/kernel/v/Read/ReadVariableOp.cond_1/Adam/dense_2/bias/v/Read/ReadVariableOp0cond_1/Adam/dense_3/kernel/v/Read/ReadVariableOp.cond_1/Adam/dense_3/bias/v/Read/ReadVariableOp/cond_1/Adam/conv1d/kernel/v/Read/ReadVariableOp-cond_1/Adam/conv1d/bias/v/Read/ReadVariableOp1cond_1/Adam/conv1d_1/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv1d_1/bias/v/Read/ReadVariableOp1cond_1/Adam/conv1d_2/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv1d_2/bias/v/Read/ReadVariableOp1cond_1/Adam/conv1d_3/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv1d_3/bias/v/Read/ReadVariableOp1cond_1/Adam/conv1d_4/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv1d_4/bias/v/Read/ReadVariableOp1cond_1/Adam/conv1d_5/kernel/v/Read/ReadVariableOp/cond_1/Adam/conv1d_5/bias/v/Read/ReadVariableOpConst*T
TinM
K2I		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 **
f%R#
!__inference__traced_save_33198547
Ь
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biascond_1/Adam/itercond_1/Adam/beta_1cond_1/Adam/beta_2cond_1/Adam/decaycond_1/Adam/learning_rateconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biasconv1d_3/kernelconv1d_3/biasconv1d_4/kernelconv1d_4/biasconv1d_5/kernelconv1d_5/biascurrent_loss_scale
good_stepstotalcounttotal_1count_1cond_1/Adam/dense/kernel/mcond_1/Adam/dense/bias/mcond_1/Adam/dense_1/kernel/mcond_1/Adam/dense_1/bias/mcond_1/Adam/dense_2/kernel/mcond_1/Adam/dense_2/bias/mcond_1/Adam/dense_3/kernel/mcond_1/Adam/dense_3/bias/mcond_1/Adam/conv1d/kernel/mcond_1/Adam/conv1d/bias/mcond_1/Adam/conv1d_1/kernel/mcond_1/Adam/conv1d_1/bias/mcond_1/Adam/conv1d_2/kernel/mcond_1/Adam/conv1d_2/bias/mcond_1/Adam/conv1d_3/kernel/mcond_1/Adam/conv1d_3/bias/mcond_1/Adam/conv1d_4/kernel/mcond_1/Adam/conv1d_4/bias/mcond_1/Adam/conv1d_5/kernel/mcond_1/Adam/conv1d_5/bias/mcond_1/Adam/dense/kernel/vcond_1/Adam/dense/bias/vcond_1/Adam/dense_1/kernel/vcond_1/Adam/dense_1/bias/vcond_1/Adam/dense_2/kernel/vcond_1/Adam/dense_2/bias/vcond_1/Adam/dense_3/kernel/vcond_1/Adam/dense_3/bias/vcond_1/Adam/conv1d/kernel/vcond_1/Adam/conv1d/bias/vcond_1/Adam/conv1d_1/kernel/vcond_1/Adam/conv1d_1/bias/vcond_1/Adam/conv1d_2/kernel/vcond_1/Adam/conv1d_2/bias/vcond_1/Adam/conv1d_3/kernel/vcond_1/Adam/conv1d_3/bias/vcond_1/Adam/conv1d_4/kernel/vcond_1/Adam/conv1d_4/bias/vcond_1/Adam/conv1d_5/kernel/vcond_1/Adam/conv1d_5/bias/v*S
TinL
J2H*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *-
f(R&
$__inference__traced_restore_33198770цй
т.

J__inference_functional_1_layer_call_and_return_conditional_losses_33197110

inputs
inputs_1
left_eye_33197059
left_eye_33197061
left_eye_33197063
left_eye_33197065
left_eye_33197067
left_eye_33197069
right_eye_33197072
right_eye_33197074
right_eye_33197076
right_eye_33197078
right_eye_33197080
right_eye_33197082
dense_33197087
dense_33197089
dense_1_33197092
dense_1_33197094
dense_2_33197097
dense_2_33197099
dense_3_33197102
dense_3_33197104
identityЂ Left_eye/StatefulPartitionedCallЂ!Right_eye/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallї
 Left_eye/StatefulPartitionedCallStatefulPartitionedCallinputsleft_eye_33197059left_eye_33197061left_eye_33197063left_eye_33197065left_eye_33197067left_eye_33197069*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Left_eye_layer_call_and_return_conditional_losses_331964582"
 Left_eye/StatefulPartitionedCall
!Right_eye/StatefulPartitionedCallStatefulPartitionedCallinputs_1right_eye_33197072right_eye_33197074right_eye_33197076right_eye_33197078right_eye_33197080right_eye_33197082*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_Right_eye_layer_call_and_return_conditional_losses_331966852#
!Right_eye/StatefulPartitionedCallИ
concatenate/PartitionedCallPartitionedCall)Left_eye/StatefulPartitionedCall:output:0*Right_eye/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_331968442
concatenate/PartitionedCallї
flatten/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_331968592
flatten/PartitionedCallЋ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_33197087dense_33197089*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_331968802
dense/StatefulPartitionedCallЛ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_33197092dense_1_33197094*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_331969092!
dense_1/StatefulPartitionedCallМ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_33197097dense_2_33197099*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_331969382!
dense_2/StatefulPartitionedCallМ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_33197102dense_3_33197104*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_331969662!
dense_3/StatefulPartitionedCall
CastCast(dense_3/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2
Castу
activation/PartitionedCallPartitionedCallCast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_331969872
activation/PartitionedCallФ
IdentityIdentity#activation/PartitionedCall:output:0!^Left_eye/StatefulPartitionedCall"^Right_eye/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd:џџџџџџџџџd::::::::::::::::::::2D
 Left_eye/StatefulPartitionedCall Left_eye/StatefulPartitionedCall2F
!Right_eye/StatefulPartitionedCall!Right_eye/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs:WS
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
ш
Ћ
/__inference_functional_1_layer_call_fn_33197654
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_331971102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd:џџџџџџџџџd::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
inputs/1
 
Л
F__inference_conv1d_5_layer_call_and_return_conditional_losses_33196623

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpЄ
conv1d/ExpandDims_1/CastCast*conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
: 2
conv1d/ExpandDims_1/Castt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЉ
conv1d/ExpandDims_1
ExpandDimsconv1d/ExpandDims_1/Cast:y:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
BiasAdd/Cast~
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ :::S O
+
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
і

+__inference_conv1d_4_layer_call_fn_33198283

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_4_layer_call_and_return_conditional_losses_331965892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ/@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ/@
 
_user_specified_nameinputs

О
,__inference_Right_eye_layer_call_fn_33196700
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_Right_eye_layer_call_and_return_conditional_losses_331966852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџd
!
_user_specified_name	input_2
Е
H
,__inference_reshape_1_layer_call_fn_33198229

inputs
identityЮ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_331965292
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd:W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
 
Л
F__inference_conv1d_3_layer_call_and_return_conditional_losses_33196555

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd	2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpЄ
conv1d/ExpandDims_1/CastCast*conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:	@2
conv1d/ExpandDims_1/Castt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЉ
conv1d/ExpandDims_1
ExpandDimsconv1d/ExpandDims_1/Cast:y:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
BiasAdd/Cast~
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџd	:::S O
+
_output_shapes
:џџџџџџџџџd	
 
_user_specified_nameinputs


­
E__inference_dense_2_layer_call_and_return_conditional_losses_33198073

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOpz
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	@2
MatMul/Caste
MatMulMatMulinputsMatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
BiasAdd/Casts
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
у

*__inference_dense_3_layer_call_fn_33198103

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_331969662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
 
Л
F__inference_conv1d_4_layer_call_and_return_conditional_losses_33198274

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpЄ
conv1d/ExpandDims_1/CastCast*conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@ 2
conv1d/ExpandDims_1/Castt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЉ
conv1d/ExpandDims_1
ExpandDimsconv1d/ExpandDims_1/Cast:y:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
BiasAdd/Cast~
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ/@:::S O
+
_output_shapes
:џџџџџџџџџ/@
 
_user_specified_nameinputs
фР
З	
J__inference_functional_1_layer_call_and_return_conditional_losses_33197608
inputs_0
inputs_1?
;left_eye_conv1d_conv1d_expanddims_1_readvariableop_resource3
/left_eye_conv1d_biasadd_readvariableop_resourceA
=left_eye_conv1d_1_conv1d_expanddims_1_readvariableop_resource5
1left_eye_conv1d_1_biasadd_readvariableop_resourceA
=left_eye_conv1d_2_conv1d_expanddims_1_readvariableop_resource5
1left_eye_conv1d_2_biasadd_readvariableop_resourceB
>right_eye_conv1d_3_conv1d_expanddims_1_readvariableop_resource6
2right_eye_conv1d_3_biasadd_readvariableop_resourceB
>right_eye_conv1d_4_conv1d_expanddims_1_readvariableop_resource6
2right_eye_conv1d_4_biasadd_readvariableop_resourceB
>right_eye_conv1d_5_conv1d_expanddims_1_readvariableop_resource6
2right_eye_conv1d_5_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityy
Left_eye/CastCastinputs_0*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
Left_eye/Castq
Left_eye/reshape/ShapeShapeLeft_eye/Cast:y:0*
T0*
_output_shapes
:2
Left_eye/reshape/Shape
$Left_eye/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Left_eye/reshape/strided_slice/stack
&Left_eye/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Left_eye/reshape/strided_slice/stack_1
&Left_eye/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Left_eye/reshape/strided_slice/stack_2Ш
Left_eye/reshape/strided_sliceStridedSliceLeft_eye/reshape/Shape:output:0-Left_eye/reshape/strided_slice/stack:output:0/Left_eye/reshape/strided_slice/stack_1:output:0/Left_eye/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Left_eye/reshape/strided_slice
 Left_eye/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2"
 Left_eye/reshape/Reshape/shape/1
 Left_eye/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2"
 Left_eye/reshape/Reshape/shape/2ѕ
Left_eye/reshape/Reshape/shapePack'Left_eye/reshape/strided_slice:output:0)Left_eye/reshape/Reshape/shape/1:output:0)Left_eye/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2 
Left_eye/reshape/Reshape/shapeБ
Left_eye/reshape/ReshapeReshapeLeft_eye/Cast:y:0'Left_eye/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2
Left_eye/reshape/Reshape
%Left_eye/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2'
%Left_eye/conv1d/conv1d/ExpandDims/dimс
!Left_eye/conv1d/conv1d/ExpandDims
ExpandDims!Left_eye/reshape/Reshape:output:0.Left_eye/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd	2#
!Left_eye/conv1d/conv1d/ExpandDimsш
2Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;left_eye_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype024
2Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOpд
(Left_eye/conv1d/conv1d/ExpandDims_1/CastCast:Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:	@2*
(Left_eye/conv1d/conv1d/ExpandDims_1/Cast
'Left_eye/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'Left_eye/conv1d/conv1d/ExpandDims_1/dimщ
#Left_eye/conv1d/conv1d/ExpandDims_1
ExpandDims,Left_eye/conv1d/conv1d/ExpandDims_1/Cast:y:00Left_eye/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2%
#Left_eye/conv1d/conv1d/ExpandDims_1ї
Left_eye/conv1d/conv1dConv2D*Left_eye/conv1d/conv1d/ExpandDims:output:0,Left_eye/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@*
paddingVALID*
strides
2
Left_eye/conv1d/conv1dТ
Left_eye/conv1d/conv1d/SqueezeSqueezeLeft_eye/conv1d/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@*
squeeze_dims

§џџџџџџџџ2 
Left_eye/conv1d/conv1d/SqueezeМ
&Left_eye/conv1d/BiasAdd/ReadVariableOpReadVariableOp/left_eye_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&Left_eye/conv1d/BiasAdd/ReadVariableOpЈ
Left_eye/conv1d/BiasAdd/CastCast.Left_eye/conv1d/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
Left_eye/conv1d/BiasAdd/CastО
Left_eye/conv1d/BiasAddBiasAdd'Left_eye/conv1d/conv1d/Squeeze:output:0 Left_eye/conv1d/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
Left_eye/conv1d/BiasAdd
Left_eye/conv1d/ReluRelu Left_eye/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
Left_eye/conv1d/Relu
'Left_eye/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2)
'Left_eye/conv1d_1/conv1d/ExpandDims/dimш
#Left_eye/conv1d_1/conv1d/ExpandDims
ExpandDims"Left_eye/conv1d/Relu:activations:00Left_eye/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@2%
#Left_eye/conv1d_1/conv1d/ExpandDimsю
4Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=left_eye_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype026
4Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpк
*Left_eye/conv1d_1/conv1d/ExpandDims_1/CastCast<Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@ 2,
*Left_eye/conv1d_1/conv1d/ExpandDims_1/Cast
)Left_eye/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)Left_eye/conv1d_1/conv1d/ExpandDims_1/dimё
%Left_eye/conv1d_1/conv1d/ExpandDims_1
ExpandDims.Left_eye/conv1d_1/conv1d/ExpandDims_1/Cast:y:02Left_eye/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2'
%Left_eye/conv1d_1/conv1d/ExpandDims_1џ
Left_eye/conv1d_1/conv1dConv2D,Left_eye/conv1d_1/conv1d/ExpandDims:output:0.Left_eye/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
Left_eye/conv1d_1/conv1dШ
 Left_eye/conv1d_1/conv1d/SqueezeSqueeze!Left_eye/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџ2"
 Left_eye/conv1d_1/conv1d/SqueezeТ
(Left_eye/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp1left_eye_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Left_eye/conv1d_1/BiasAdd/ReadVariableOpЎ
Left_eye/conv1d_1/BiasAdd/CastCast0Left_eye/conv1d_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2 
Left_eye/conv1d_1/BiasAdd/CastЦ
Left_eye/conv1d_1/BiasAddBiasAdd)Left_eye/conv1d_1/conv1d/Squeeze:output:0"Left_eye/conv1d_1/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
Left_eye/conv1d_1/BiasAdd
Left_eye/conv1d_1/ReluRelu"Left_eye/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
Left_eye/conv1d_1/Relu
'Left_eye/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2)
'Left_eye/conv1d_2/conv1d/ExpandDims/dimъ
#Left_eye/conv1d_2/conv1d/ExpandDims
ExpandDims$Left_eye/conv1d_1/Relu:activations:00Left_eye/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2%
#Left_eye/conv1d_2/conv1d/ExpandDimsю
4Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=left_eye_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype026
4Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpк
*Left_eye/conv1d_2/conv1d/ExpandDims_1/CastCast<Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
: 2,
*Left_eye/conv1d_2/conv1d/ExpandDims_1/Cast
)Left_eye/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)Left_eye/conv1d_2/conv1d/ExpandDims_1/dimё
%Left_eye/conv1d_2/conv1d/ExpandDims_1
ExpandDims.Left_eye/conv1d_2/conv1d/ExpandDims_1/Cast:y:02Left_eye/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2'
%Left_eye/conv1d_2/conv1d/ExpandDims_1џ
Left_eye/conv1d_2/conv1dConv2D,Left_eye/conv1d_2/conv1d/ExpandDims:output:0.Left_eye/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
*
paddingVALID*
strides
2
Left_eye/conv1d_2/conv1dШ
 Left_eye/conv1d_2/conv1d/SqueezeSqueeze!Left_eye/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
*
squeeze_dims

§џџџџџџџџ2"
 Left_eye/conv1d_2/conv1d/SqueezeТ
(Left_eye/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp1left_eye_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Left_eye/conv1d_2/BiasAdd/ReadVariableOpЎ
Left_eye/conv1d_2/BiasAdd/CastCast0Left_eye/conv1d_2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2 
Left_eye/conv1d_2/BiasAdd/CastЦ
Left_eye/conv1d_2/BiasAddBiasAdd)Left_eye/conv1d_2/conv1d/Squeeze:output:0"Left_eye/conv1d_2/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
Left_eye/conv1d_2/BiasAdd
Left_eye/conv1d_2/ReluRelu"Left_eye/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
Left_eye/conv1d_2/Relu{
Right_eye/CastCastinputs_1*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
Right_eye/Castx
Right_eye/reshape_1/ShapeShapeRight_eye/Cast:y:0*
T0*
_output_shapes
:2
Right_eye/reshape_1/Shape
'Right_eye/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'Right_eye/reshape_1/strided_slice/stack 
)Right_eye/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)Right_eye/reshape_1/strided_slice/stack_1 
)Right_eye/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)Right_eye/reshape_1/strided_slice/stack_2к
!Right_eye/reshape_1/strided_sliceStridedSlice"Right_eye/reshape_1/Shape:output:00Right_eye/reshape_1/strided_slice/stack:output:02Right_eye/reshape_1/strided_slice/stack_1:output:02Right_eye/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!Right_eye/reshape_1/strided_slice
#Right_eye/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2%
#Right_eye/reshape_1/Reshape/shape/1
#Right_eye/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2%
#Right_eye/reshape_1/Reshape/shape/2
!Right_eye/reshape_1/Reshape/shapePack*Right_eye/reshape_1/strided_slice:output:0,Right_eye/reshape_1/Reshape/shape/1:output:0,Right_eye/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!Right_eye/reshape_1/Reshape/shapeЛ
Right_eye/reshape_1/ReshapeReshapeRight_eye/Cast:y:0*Right_eye/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2
Right_eye/reshape_1/Reshape
(Right_eye/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(Right_eye/conv1d_3/conv1d/ExpandDims/dimэ
$Right_eye/conv1d_3/conv1d/ExpandDims
ExpandDims$Right_eye/reshape_1/Reshape:output:01Right_eye/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd	2&
$Right_eye/conv1d_3/conv1d/ExpandDimsё
5Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>right_eye_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype027
5Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpн
+Right_eye/conv1d_3/conv1d/ExpandDims_1/CastCast=Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:	@2-
+Right_eye/conv1d_3/conv1d/ExpandDims_1/Cast
*Right_eye/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*Right_eye/conv1d_3/conv1d/ExpandDims_1/dimѕ
&Right_eye/conv1d_3/conv1d/ExpandDims_1
ExpandDims/Right_eye/conv1d_3/conv1d/ExpandDims_1/Cast:y:03Right_eye/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2(
&Right_eye/conv1d_3/conv1d/ExpandDims_1
Right_eye/conv1d_3/conv1dConv2D-Right_eye/conv1d_3/conv1d/ExpandDims:output:0/Right_eye/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@*
paddingVALID*
strides
2
Right_eye/conv1d_3/conv1dЫ
!Right_eye/conv1d_3/conv1d/SqueezeSqueeze"Right_eye/conv1d_3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@*
squeeze_dims

§џџџџџџџџ2#
!Right_eye/conv1d_3/conv1d/SqueezeХ
)Right_eye/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp2right_eye_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)Right_eye/conv1d_3/BiasAdd/ReadVariableOpБ
Right_eye/conv1d_3/BiasAdd/CastCast1Right_eye/conv1d_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2!
Right_eye/conv1d_3/BiasAdd/CastЪ
Right_eye/conv1d_3/BiasAddBiasAdd*Right_eye/conv1d_3/conv1d/Squeeze:output:0#Right_eye/conv1d_3/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
Right_eye/conv1d_3/BiasAdd
Right_eye/conv1d_3/ReluRelu#Right_eye/conv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
Right_eye/conv1d_3/Relu
(Right_eye/conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(Right_eye/conv1d_4/conv1d/ExpandDims/dimю
$Right_eye/conv1d_4/conv1d/ExpandDims
ExpandDims%Right_eye/conv1d_3/Relu:activations:01Right_eye/conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@2&
$Right_eye/conv1d_4/conv1d/ExpandDimsё
5Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>right_eye_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype027
5Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpн
+Right_eye/conv1d_4/conv1d/ExpandDims_1/CastCast=Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@ 2-
+Right_eye/conv1d_4/conv1d/ExpandDims_1/Cast
*Right_eye/conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*Right_eye/conv1d_4/conv1d/ExpandDims_1/dimѕ
&Right_eye/conv1d_4/conv1d/ExpandDims_1
ExpandDims/Right_eye/conv1d_4/conv1d/ExpandDims_1/Cast:y:03Right_eye/conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2(
&Right_eye/conv1d_4/conv1d/ExpandDims_1
Right_eye/conv1d_4/conv1dConv2D-Right_eye/conv1d_4/conv1d/ExpandDims:output:0/Right_eye/conv1d_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
Right_eye/conv1d_4/conv1dЫ
!Right_eye/conv1d_4/conv1d/SqueezeSqueeze"Right_eye/conv1d_4/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџ2#
!Right_eye/conv1d_4/conv1d/SqueezeХ
)Right_eye/conv1d_4/BiasAdd/ReadVariableOpReadVariableOp2right_eye_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)Right_eye/conv1d_4/BiasAdd/ReadVariableOpБ
Right_eye/conv1d_4/BiasAdd/CastCast1Right_eye/conv1d_4/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2!
Right_eye/conv1d_4/BiasAdd/CastЪ
Right_eye/conv1d_4/BiasAddBiasAdd*Right_eye/conv1d_4/conv1d/Squeeze:output:0#Right_eye/conv1d_4/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
Right_eye/conv1d_4/BiasAdd
Right_eye/conv1d_4/ReluRelu#Right_eye/conv1d_4/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
Right_eye/conv1d_4/Relu
(Right_eye/conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(Right_eye/conv1d_5/conv1d/ExpandDims/dimю
$Right_eye/conv1d_5/conv1d/ExpandDims
ExpandDims%Right_eye/conv1d_4/Relu:activations:01Right_eye/conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2&
$Right_eye/conv1d_5/conv1d/ExpandDimsё
5Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>right_eye_conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpн
+Right_eye/conv1d_5/conv1d/ExpandDims_1/CastCast=Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
: 2-
+Right_eye/conv1d_5/conv1d/ExpandDims_1/Cast
*Right_eye/conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*Right_eye/conv1d_5/conv1d/ExpandDims_1/dimѕ
&Right_eye/conv1d_5/conv1d/ExpandDims_1
ExpandDims/Right_eye/conv1d_5/conv1d/ExpandDims_1/Cast:y:03Right_eye/conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2(
&Right_eye/conv1d_5/conv1d/ExpandDims_1
Right_eye/conv1d_5/conv1dConv2D-Right_eye/conv1d_5/conv1d/ExpandDims:output:0/Right_eye/conv1d_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
*
paddingVALID*
strides
2
Right_eye/conv1d_5/conv1dЫ
!Right_eye/conv1d_5/conv1d/SqueezeSqueeze"Right_eye/conv1d_5/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
*
squeeze_dims

§џџџџџџџџ2#
!Right_eye/conv1d_5/conv1d/SqueezeХ
)Right_eye/conv1d_5/BiasAdd/ReadVariableOpReadVariableOp2right_eye_conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)Right_eye/conv1d_5/BiasAdd/ReadVariableOpБ
Right_eye/conv1d_5/BiasAdd/CastCast1Right_eye/conv1d_5/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2!
Right_eye/conv1d_5/BiasAdd/CastЪ
Right_eye/conv1d_5/BiasAddBiasAdd*Right_eye/conv1d_5/conv1d/Squeeze:output:0#Right_eye/conv1d_5/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
Right_eye/conv1d_5/BiasAdd
Right_eye/conv1d_5/ReluRelu#Right_eye/conv1d_5/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
Right_eye/conv1d_5/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisт
concatenate/concatConcatV2$Left_eye/conv1d_2/Relu:activations:0%Right_eye/conv1d_5/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ
 2
concatenate/concato
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
flatten/Const
flatten/ReshapeReshapeconcatenate/concat:output:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
flatten/ReshapeЁ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
Р*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMul/CastCast#dense/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:
Р2
dense/MatMul/Cast
dense/MatMulMatMulflatten/Reshape:output:0dense/MatMul/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAdd/CastCast$dense/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:2
dense/BiasAdd/Cast
dense/BiasAddBiasAdddense/MatMul:product:0dense/BiasAdd/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

dense/ReluЇ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMul/CastCast%dense_1/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:
2
dense_1/MatMul/Cast
dense_1/MatMulMatMuldense/Relu:activations:0dense_1/MatMul/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_1/MatMulЅ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAdd/CastCast&dense_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:2
dense_1/BiasAdd/Cast
dense_1/BiasAddBiasAdddense_1/MatMul:product:0dense_1/BiasAdd/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_1/ReluІ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMul/CastCast%dense_2/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	@2
dense_2/MatMul/Cast
dense_2/MatMulMatMuldense_1/Relu:activations:0dense_2/MatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_2/MatMulЄ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAdd/CastCast&dense_2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
dense_2/BiasAdd/Cast
dense_2/BiasAddBiasAdddense_2/MatMul:product:0dense_2/BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_2/ReluЅ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMul/CastCast%dense_3/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes

:@2
dense_3/MatMul/Cast
dense_3/MatMulMatMuldense_2/Relu:activations:0dense_3/MatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/MatMulЄ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp
dense_3/BiasAdd/CastCast&dense_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
dense_3/BiasAdd/Cast
dense_3/BiasAddBiasAdddense_3/MatMul:product:0dense_3/BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/BiasAddo
CastCastdense_3/BiasAdd:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2
Cast\
IdentityIdentityCast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd:џџџџџџџџџd:::::::::::::::::::::Y U
/
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
inputs/1
№
c
G__inference_reshape_1_layer_call_and_return_conditional_losses_33198224

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd:W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
е.

J__inference_functional_1_layer_call_and_return_conditional_losses_33197051
left	
right
left_eye_33197000
left_eye_33197002
left_eye_33197004
left_eye_33197006
left_eye_33197008
left_eye_33197010
right_eye_33197013
right_eye_33197015
right_eye_33197017
right_eye_33197019
right_eye_33197021
right_eye_33197023
dense_33197028
dense_33197030
dense_1_33197033
dense_1_33197035
dense_2_33197038
dense_2_33197040
dense_3_33197043
dense_3_33197045
identityЂ Left_eye/StatefulPartitionedCallЂ!Right_eye/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallѕ
 Left_eye/StatefulPartitionedCallStatefulPartitionedCallleftleft_eye_33197000left_eye_33197002left_eye_33197004left_eye_33197006left_eye_33197008left_eye_33197010*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Left_eye_layer_call_and_return_conditional_losses_331964962"
 Left_eye/StatefulPartitionedCallџ
!Right_eye/StatefulPartitionedCallStatefulPartitionedCallrightright_eye_33197013right_eye_33197015right_eye_33197017right_eye_33197019right_eye_33197021right_eye_33197023*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_Right_eye_layer_call_and_return_conditional_losses_331967232#
!Right_eye/StatefulPartitionedCallИ
concatenate/PartitionedCallPartitionedCall)Left_eye/StatefulPartitionedCall:output:0*Right_eye/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_331968442
concatenate/PartitionedCallї
flatten/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_331968592
flatten/PartitionedCallЋ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_33197028dense_33197030*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_331968802
dense/StatefulPartitionedCallЛ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_33197033dense_1_33197035*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_331969092!
dense_1/StatefulPartitionedCallМ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_33197038dense_2_33197040*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_331969382!
dense_2/StatefulPartitionedCallМ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_33197043dense_3_33197045*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_331969662!
dense_3/StatefulPartitionedCall
CastCast(dense_3/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2
Castу
activation/PartitionedCallPartitionedCallCast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_331969872
activation/PartitionedCallФ
IdentityIdentity#activation/PartitionedCall:output:0!^Left_eye/StatefulPartitionedCall"^Right_eye/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd:џџџџџџџџџd::::::::::::::::::::2D
 Left_eye/StatefulPartitionedCall Left_eye/StatefulPartitionedCall2F
!Right_eye/StatefulPartitionedCall!Right_eye/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:U Q
/
_output_shapes
:џџџџџџџџџd

_user_specified_nameLeft:VR
/
_output_shapes
:џџџџџџџџџd

_user_specified_nameRight
п
г
F__inference_Left_eye_layer_call_and_return_conditional_losses_33196434
input_1
conv1d_33196418
conv1d_33196420
conv1d_1_33196423
conv1d_1_33196425
conv1d_2_33196428
conv1d_2_33196430
identityЂconv1d/StatefulPartitionedCallЂ conv1d_1/StatefulPartitionedCallЂ conv1d_2/StatefulPartitionedCallf
CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
Castо
reshape/PartitionedCallPartitionedCallCast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_331963022
reshape/PartitionedCallГ
conv1d/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1d_33196418conv1d_33196420*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv1d_layer_call_and_return_conditional_losses_331963282 
conv1d/StatefulPartitionedCallФ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_33196423conv1d_1_33196425*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_1_layer_call_and_return_conditional_losses_331963622"
 conv1d_1/StatefulPartitionedCallЦ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_2_33196428conv1d_2_33196430*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_331963962"
 conv1d_2/StatefulPartitionedCallш
IdentityIdentity)conv1d_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџd
!
_user_specified_name	input_1
 
Л
F__inference_conv1d_1_layer_call_and_return_conditional_losses_33196362

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpЄ
conv1d/ExpandDims_1/CastCast*conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@ 2
conv1d/ExpandDims_1/Castt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЉ
conv1d/ExpandDims_1
ExpandDimsconv1d/ExpandDims_1/Cast:y:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
BiasAdd/Cast~
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ/@:::S O
+
_output_shapes
:џџџџџџџџџ/@
 
_user_specified_nameinputs
нB

G__inference_Right_eye_layer_call_and_return_conditional_losses_33197958

inputs8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource8
4conv1d_4_conv1d_expanddims_1_readvariableop_resource,
(conv1d_4_biasadd_readvariableop_resource8
4conv1d_5_conv1d_expanddims_1_readvariableop_resource,
(conv1d_5_biasadd_readvariableop_resource
identitye
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
CastZ
reshape_1/ShapeShapeCast:y:0*
T0*
_output_shapes
:2
reshape_1/Shape
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape_1/Reshape/shape/2в
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape
reshape_1/ReshapeReshapeCast:y:0 reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2
reshape_1/Reshape
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
conv1d_3/conv1d/ExpandDims/dimХ
conv1d_3/conv1d/ExpandDims
ExpandDimsreshape_1/Reshape:output:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd	2
conv1d_3/conv1d/ExpandDimsг
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpП
!conv1d_3/conv1d/ExpandDims_1/CastCast3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:	@2#
!conv1d_3/conv1d/ExpandDims_1/Cast
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dimЭ
conv1d_3/conv1d/ExpandDims_1
ExpandDims%conv1d_3/conv1d/ExpandDims_1/Cast:y:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d_3/conv1d/ExpandDims_1л
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@*
paddingVALID*
strides
2
conv1d_3/conv1d­
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@*
squeeze_dims

§џџџџџџџџ2
conv1d_3/conv1d/SqueezeЇ
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_3/BiasAdd/ReadVariableOp
conv1d_3/BiasAdd/CastCast'conv1d_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
conv1d_3/BiasAdd/CastЂ
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0conv1d_3/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
conv1d_3/BiasAddw
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
conv1d_3/Relu
conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
conv1d_4/conv1d/ExpandDims/dimЦ
conv1d_4/conv1d/ExpandDims
ExpandDimsconv1d_3/Relu:activations:0'conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@2
conv1d_4/conv1d/ExpandDimsг
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02-
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpП
!conv1d_4/conv1d/ExpandDims_1/CastCast3conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@ 2#
!conv1d_4/conv1d/ExpandDims_1/Cast
 conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_4/conv1d/ExpandDims_1/dimЭ
conv1d_4/conv1d/ExpandDims_1
ExpandDims%conv1d_4/conv1d/ExpandDims_1/Cast:y:0)conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d_4/conv1d/ExpandDims_1л
conv1d_4/conv1dConv2D#conv1d_4/conv1d/ExpandDims:output:0%conv1d_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv1d_4/conv1d­
conv1d_4/conv1d/SqueezeSqueezeconv1d_4/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџ2
conv1d_4/conv1d/SqueezeЇ
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_4/BiasAdd/ReadVariableOp
conv1d_4/BiasAdd/CastCast'conv1d_4/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
conv1d_4/BiasAdd/CastЂ
conv1d_4/BiasAddBiasAdd conv1d_4/conv1d/Squeeze:output:0conv1d_4/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
conv1d_4/BiasAddw
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
conv1d_4/Relu
conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
conv1d_5/conv1d/ExpandDims/dimЦ
conv1d_5/conv1d/ExpandDims
ExpandDimsconv1d_4/Relu:activations:0'conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv1d_5/conv1d/ExpandDimsг
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpП
!conv1d_5/conv1d/ExpandDims_1/CastCast3conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
: 2#
!conv1d_5/conv1d/ExpandDims_1/Cast
 conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_5/conv1d/ExpandDims_1/dimЭ
conv1d_5/conv1d/ExpandDims_1
ExpandDims%conv1d_5/conv1d/ExpandDims_1/Cast:y:0)conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_5/conv1d/ExpandDims_1л
conv1d_5/conv1dConv2D#conv1d_5/conv1d/ExpandDims:output:0%conv1d_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
*
paddingVALID*
strides
2
conv1d_5/conv1d­
conv1d_5/conv1d/SqueezeSqueezeconv1d_5/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
*
squeeze_dims

§џџџџџџџџ2
conv1d_5/conv1d/SqueezeЇ
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_5/BiasAdd/ReadVariableOp
conv1d_5/BiasAdd/CastCast'conv1d_5/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
conv1d_5/BiasAdd/CastЂ
conv1d_5/BiasAddBiasAdd conv1d_5/conv1d/Squeeze:output:0conv1d_5/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
conv1d_5/BiasAddw
conv1d_5/ReluReluconv1d_5/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
conv1d_5/Relus
IdentityIdentityconv1d_5/Relu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd:::::::W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
О
Z
.__inference_concatenate_layer_call_fn_33198005
inputs_0
inputs_1
identityн
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_331968442
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:џџџџџџџџџ
:џџџџџџџџџ
:U Q
+
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
inputs/1


Ћ
C__inference_dense_layer_call_and_return_conditional_losses_33196880

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Р*
dtype02
MatMul/ReadVariableOp{
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:
Р2
MatMul/Castf
MatMulMatMulinputsMatMul/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:2
BiasAdd/Castt
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР:::P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
Б
F
*__inference_reshape_layer_call_fn_33198130

inputs
identityЬ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_331963022
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd:W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Иъ

#__inference__wrapped_model_33196284
left	
rightL
Hfunctional_1_left_eye_conv1d_conv1d_expanddims_1_readvariableop_resource@
<functional_1_left_eye_conv1d_biasadd_readvariableop_resourceN
Jfunctional_1_left_eye_conv1d_1_conv1d_expanddims_1_readvariableop_resourceB
>functional_1_left_eye_conv1d_1_biasadd_readvariableop_resourceN
Jfunctional_1_left_eye_conv1d_2_conv1d_expanddims_1_readvariableop_resourceB
>functional_1_left_eye_conv1d_2_biasadd_readvariableop_resourceO
Kfunctional_1_right_eye_conv1d_3_conv1d_expanddims_1_readvariableop_resourceC
?functional_1_right_eye_conv1d_3_biasadd_readvariableop_resourceO
Kfunctional_1_right_eye_conv1d_4_conv1d_expanddims_1_readvariableop_resourceC
?functional_1_right_eye_conv1d_4_biasadd_readvariableop_resourceO
Kfunctional_1_right_eye_conv1d_5_conv1d_expanddims_1_readvariableop_resourceC
?functional_1_right_eye_conv1d_5_biasadd_readvariableop_resource5
1functional_1_dense_matmul_readvariableop_resource6
2functional_1_dense_biasadd_readvariableop_resource7
3functional_1_dense_1_matmul_readvariableop_resource8
4functional_1_dense_1_biasadd_readvariableop_resource7
3functional_1_dense_2_matmul_readvariableop_resource8
4functional_1_dense_2_biasadd_readvariableop_resource7
3functional_1_dense_3_matmul_readvariableop_resource8
4functional_1_dense_3_biasadd_readvariableop_resource
identity
functional_1/Left_eye/CastCastleft*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
functional_1/Left_eye/Cast
#functional_1/Left_eye/reshape/ShapeShapefunctional_1/Left_eye/Cast:y:0*
T0*
_output_shapes
:2%
#functional_1/Left_eye/reshape/ShapeА
1functional_1/Left_eye/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1functional_1/Left_eye/reshape/strided_slice/stackД
3functional_1/Left_eye/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_1/Left_eye/reshape/strided_slice/stack_1Д
3functional_1/Left_eye/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_1/Left_eye/reshape/strided_slice/stack_2
+functional_1/Left_eye/reshape/strided_sliceStridedSlice,functional_1/Left_eye/reshape/Shape:output:0:functional_1/Left_eye/reshape/strided_slice/stack:output:0<functional_1/Left_eye/reshape/strided_slice/stack_1:output:0<functional_1/Left_eye/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+functional_1/Left_eye/reshape/strided_slice 
-functional_1/Left_eye/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2/
-functional_1/Left_eye/reshape/Reshape/shape/1 
-functional_1/Left_eye/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2/
-functional_1/Left_eye/reshape/Reshape/shape/2Ж
+functional_1/Left_eye/reshape/Reshape/shapePack4functional_1/Left_eye/reshape/strided_slice:output:06functional_1/Left_eye/reshape/Reshape/shape/1:output:06functional_1/Left_eye/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2-
+functional_1/Left_eye/reshape/Reshape/shapeх
%functional_1/Left_eye/reshape/ReshapeReshapefunctional_1/Left_eye/Cast:y:04functional_1/Left_eye/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2'
%functional_1/Left_eye/reshape/ReshapeГ
2functional_1/Left_eye/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ24
2functional_1/Left_eye/conv1d/conv1d/ExpandDims/dim
.functional_1/Left_eye/conv1d/conv1d/ExpandDims
ExpandDims.functional_1/Left_eye/reshape/Reshape:output:0;functional_1/Left_eye/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd	20
.functional_1/Left_eye/conv1d/conv1d/ExpandDims
?functional_1/Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHfunctional_1_left_eye_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02A
?functional_1/Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOpћ
5functional_1/Left_eye/conv1d/conv1d/ExpandDims_1/CastCastGfunctional_1/Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:	@27
5functional_1/Left_eye/conv1d/conv1d/ExpandDims_1/CastЎ
4functional_1/Left_eye/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4functional_1/Left_eye/conv1d/conv1d/ExpandDims_1/dim
0functional_1/Left_eye/conv1d/conv1d/ExpandDims_1
ExpandDims9functional_1/Left_eye/conv1d/conv1d/ExpandDims_1/Cast:y:0=functional_1/Left_eye/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@22
0functional_1/Left_eye/conv1d/conv1d/ExpandDims_1Ћ
#functional_1/Left_eye/conv1d/conv1dConv2D7functional_1/Left_eye/conv1d/conv1d/ExpandDims:output:09functional_1/Left_eye/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@*
paddingVALID*
strides
2%
#functional_1/Left_eye/conv1d/conv1dщ
+functional_1/Left_eye/conv1d/conv1d/SqueezeSqueeze,functional_1/Left_eye/conv1d/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@*
squeeze_dims

§џџџџџџџџ2-
+functional_1/Left_eye/conv1d/conv1d/Squeezeу
3functional_1/Left_eye/conv1d/BiasAdd/ReadVariableOpReadVariableOp<functional_1_left_eye_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3functional_1/Left_eye/conv1d/BiasAdd/ReadVariableOpЯ
)functional_1/Left_eye/conv1d/BiasAdd/CastCast;functional_1/Left_eye/conv1d/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2+
)functional_1/Left_eye/conv1d/BiasAdd/Castђ
$functional_1/Left_eye/conv1d/BiasAddBiasAdd4functional_1/Left_eye/conv1d/conv1d/Squeeze:output:0-functional_1/Left_eye/conv1d/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2&
$functional_1/Left_eye/conv1d/BiasAddГ
!functional_1/Left_eye/conv1d/ReluRelu-functional_1/Left_eye/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2#
!functional_1/Left_eye/conv1d/ReluЗ
4functional_1/Left_eye/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ26
4functional_1/Left_eye/conv1d_1/conv1d/ExpandDims/dim
0functional_1/Left_eye/conv1d_1/conv1d/ExpandDims
ExpandDims/functional_1/Left_eye/conv1d/Relu:activations:0=functional_1/Left_eye/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@22
0functional_1/Left_eye/conv1d_1/conv1d/ExpandDims
Afunctional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpJfunctional_1_left_eye_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02C
Afunctional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp
7functional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1/CastCastIfunctional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@ 29
7functional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1/CastВ
6functional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 28
6functional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1/dimЅ
2functional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1
ExpandDims;functional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1/Cast:y:0?functional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 24
2functional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1Г
%functional_1/Left_eye/conv1d_1/conv1dConv2D9functional_1/Left_eye/conv1d_1/conv1d/ExpandDims:output:0;functional_1/Left_eye/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2'
%functional_1/Left_eye/conv1d_1/conv1dя
-functional_1/Left_eye/conv1d_1/conv1d/SqueezeSqueeze.functional_1/Left_eye/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџ2/
-functional_1/Left_eye/conv1d_1/conv1d/Squeezeщ
5functional_1/Left_eye/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp>functional_1_left_eye_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5functional_1/Left_eye/conv1d_1/BiasAdd/ReadVariableOpе
+functional_1/Left_eye/conv1d_1/BiasAdd/CastCast=functional_1/Left_eye/conv1d_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+functional_1/Left_eye/conv1d_1/BiasAdd/Castњ
&functional_1/Left_eye/conv1d_1/BiasAddBiasAdd6functional_1/Left_eye/conv1d_1/conv1d/Squeeze:output:0/functional_1/Left_eye/conv1d_1/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ 2(
&functional_1/Left_eye/conv1d_1/BiasAddЙ
#functional_1/Left_eye/conv1d_1/ReluRelu/functional_1/Left_eye/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2%
#functional_1/Left_eye/conv1d_1/ReluЗ
4functional_1/Left_eye/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ26
4functional_1/Left_eye/conv1d_2/conv1d/ExpandDims/dim
0functional_1/Left_eye/conv1d_2/conv1d/ExpandDims
ExpandDims1functional_1/Left_eye/conv1d_1/Relu:activations:0=functional_1/Left_eye/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 22
0functional_1/Left_eye/conv1d_2/conv1d/ExpandDims
Afunctional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpJfunctional_1_left_eye_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02C
Afunctional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp
7functional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1/CastCastIfunctional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
: 29
7functional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1/CastВ
6functional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 28
6functional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1/dimЅ
2functional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1
ExpandDims;functional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1/Cast:y:0?functional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 24
2functional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1Г
%functional_1/Left_eye/conv1d_2/conv1dConv2D9functional_1/Left_eye/conv1d_2/conv1d/ExpandDims:output:0;functional_1/Left_eye/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
*
paddingVALID*
strides
2'
%functional_1/Left_eye/conv1d_2/conv1dя
-functional_1/Left_eye/conv1d_2/conv1d/SqueezeSqueeze.functional_1/Left_eye/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
*
squeeze_dims

§џџџџџџџџ2/
-functional_1/Left_eye/conv1d_2/conv1d/Squeezeщ
5functional_1/Left_eye/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp>functional_1_left_eye_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5functional_1/Left_eye/conv1d_2/BiasAdd/ReadVariableOpе
+functional_1/Left_eye/conv1d_2/BiasAdd/CastCast=functional_1/Left_eye/conv1d_2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2-
+functional_1/Left_eye/conv1d_2/BiasAdd/Castњ
&functional_1/Left_eye/conv1d_2/BiasAddBiasAdd6functional_1/Left_eye/conv1d_2/conv1d/Squeeze:output:0/functional_1/Left_eye/conv1d_2/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
2(
&functional_1/Left_eye/conv1d_2/BiasAddЙ
#functional_1/Left_eye/conv1d_2/ReluRelu/functional_1/Left_eye/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
2%
#functional_1/Left_eye/conv1d_2/Relu
functional_1/Right_eye/CastCastright*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
functional_1/Right_eye/Cast
&functional_1/Right_eye/reshape_1/ShapeShapefunctional_1/Right_eye/Cast:y:0*
T0*
_output_shapes
:2(
&functional_1/Right_eye/reshape_1/ShapeЖ
4functional_1/Right_eye/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4functional_1/Right_eye/reshape_1/strided_slice/stackК
6functional_1/Right_eye/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6functional_1/Right_eye/reshape_1/strided_slice/stack_1К
6functional_1/Right_eye/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6functional_1/Right_eye/reshape_1/strided_slice/stack_2Ј
.functional_1/Right_eye/reshape_1/strided_sliceStridedSlice/functional_1/Right_eye/reshape_1/Shape:output:0=functional_1/Right_eye/reshape_1/strided_slice/stack:output:0?functional_1/Right_eye/reshape_1/strided_slice/stack_1:output:0?functional_1/Right_eye/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.functional_1/Right_eye/reshape_1/strided_sliceІ
0functional_1/Right_eye/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d22
0functional_1/Right_eye/reshape_1/Reshape/shape/1І
0functional_1/Right_eye/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	22
0functional_1/Right_eye/reshape_1/Reshape/shape/2Х
.functional_1/Right_eye/reshape_1/Reshape/shapePack7functional_1/Right_eye/reshape_1/strided_slice:output:09functional_1/Right_eye/reshape_1/Reshape/shape/1:output:09functional_1/Right_eye/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:20
.functional_1/Right_eye/reshape_1/Reshape/shapeя
(functional_1/Right_eye/reshape_1/ReshapeReshapefunctional_1/Right_eye/Cast:y:07functional_1/Right_eye/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2*
(functional_1/Right_eye/reshape_1/ReshapeЙ
5functional_1/Right_eye/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ27
5functional_1/Right_eye/conv1d_3/conv1d/ExpandDims/dimЁ
1functional_1/Right_eye/conv1d_3/conv1d/ExpandDims
ExpandDims1functional_1/Right_eye/reshape_1/Reshape:output:0>functional_1/Right_eye/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd	23
1functional_1/Right_eye/conv1d_3/conv1d/ExpandDims
Bfunctional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKfunctional_1_right_eye_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02D
Bfunctional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp
8functional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1/CastCastJfunctional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:	@2:
8functional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1/CastД
7functional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7functional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1/dimЉ
3functional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1
ExpandDims<functional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1/Cast:y:0@functional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@25
3functional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1З
&functional_1/Right_eye/conv1d_3/conv1dConv2D:functional_1/Right_eye/conv1d_3/conv1d/ExpandDims:output:0<functional_1/Right_eye/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@*
paddingVALID*
strides
2(
&functional_1/Right_eye/conv1d_3/conv1dђ
.functional_1/Right_eye/conv1d_3/conv1d/SqueezeSqueeze/functional_1/Right_eye/conv1d_3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@*
squeeze_dims

§џџџџџџџџ20
.functional_1/Right_eye/conv1d_3/conv1d/Squeezeь
6functional_1/Right_eye/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp?functional_1_right_eye_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6functional_1/Right_eye/conv1d_3/BiasAdd/ReadVariableOpи
,functional_1/Right_eye/conv1d_3/BiasAdd/CastCast>functional_1/Right_eye/conv1d_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2.
,functional_1/Right_eye/conv1d_3/BiasAdd/Castў
'functional_1/Right_eye/conv1d_3/BiasAddBiasAdd7functional_1/Right_eye/conv1d_3/conv1d/Squeeze:output:00functional_1/Right_eye/conv1d_3/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2)
'functional_1/Right_eye/conv1d_3/BiasAddМ
$functional_1/Right_eye/conv1d_3/ReluRelu0functional_1/Right_eye/conv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2&
$functional_1/Right_eye/conv1d_3/ReluЙ
5functional_1/Right_eye/conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ27
5functional_1/Right_eye/conv1d_4/conv1d/ExpandDims/dimЂ
1functional_1/Right_eye/conv1d_4/conv1d/ExpandDims
ExpandDims2functional_1/Right_eye/conv1d_3/Relu:activations:0>functional_1/Right_eye/conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@23
1functional_1/Right_eye/conv1d_4/conv1d/ExpandDims
Bfunctional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKfunctional_1_right_eye_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02D
Bfunctional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp
8functional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1/CastCastJfunctional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@ 2:
8functional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1/CastД
7functional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7functional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1/dimЉ
3functional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1
ExpandDims<functional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1/Cast:y:0@functional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 25
3functional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1З
&functional_1/Right_eye/conv1d_4/conv1dConv2D:functional_1/Right_eye/conv1d_4/conv1d/ExpandDims:output:0<functional_1/Right_eye/conv1d_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2(
&functional_1/Right_eye/conv1d_4/conv1dђ
.functional_1/Right_eye/conv1d_4/conv1d/SqueezeSqueeze/functional_1/Right_eye/conv1d_4/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџ20
.functional_1/Right_eye/conv1d_4/conv1d/Squeezeь
6functional_1/Right_eye/conv1d_4/BiasAdd/ReadVariableOpReadVariableOp?functional_1_right_eye_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6functional_1/Right_eye/conv1d_4/BiasAdd/ReadVariableOpи
,functional_1/Right_eye/conv1d_4/BiasAdd/CastCast>functional_1/Right_eye/conv1d_4/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,functional_1/Right_eye/conv1d_4/BiasAdd/Castў
'functional_1/Right_eye/conv1d_4/BiasAddBiasAdd7functional_1/Right_eye/conv1d_4/conv1d/Squeeze:output:00functional_1/Right_eye/conv1d_4/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ 2)
'functional_1/Right_eye/conv1d_4/BiasAddМ
$functional_1/Right_eye/conv1d_4/ReluRelu0functional_1/Right_eye/conv1d_4/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2&
$functional_1/Right_eye/conv1d_4/ReluЙ
5functional_1/Right_eye/conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ27
5functional_1/Right_eye/conv1d_5/conv1d/ExpandDims/dimЂ
1functional_1/Right_eye/conv1d_5/conv1d/ExpandDims
ExpandDims2functional_1/Right_eye/conv1d_4/Relu:activations:0>functional_1/Right_eye/conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 23
1functional_1/Right_eye/conv1d_5/conv1d/ExpandDims
Bfunctional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpKfunctional_1_right_eye_conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02D
Bfunctional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp
8functional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1/CastCastJfunctional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
: 2:
8functional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1/CastД
7functional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7functional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1/dimЉ
3functional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1
ExpandDims<functional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1/Cast:y:0@functional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 25
3functional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1З
&functional_1/Right_eye/conv1d_5/conv1dConv2D:functional_1/Right_eye/conv1d_5/conv1d/ExpandDims:output:0<functional_1/Right_eye/conv1d_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
*
paddingVALID*
strides
2(
&functional_1/Right_eye/conv1d_5/conv1dђ
.functional_1/Right_eye/conv1d_5/conv1d/SqueezeSqueeze/functional_1/Right_eye/conv1d_5/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
*
squeeze_dims

§џџџџџџџџ20
.functional_1/Right_eye/conv1d_5/conv1d/Squeezeь
6functional_1/Right_eye/conv1d_5/BiasAdd/ReadVariableOpReadVariableOp?functional_1_right_eye_conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6functional_1/Right_eye/conv1d_5/BiasAdd/ReadVariableOpи
,functional_1/Right_eye/conv1d_5/BiasAdd/CastCast>functional_1/Right_eye/conv1d_5/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2.
,functional_1/Right_eye/conv1d_5/BiasAdd/Castў
'functional_1/Right_eye/conv1d_5/BiasAddBiasAdd7functional_1/Right_eye/conv1d_5/conv1d/Squeeze:output:00functional_1/Right_eye/conv1d_5/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
2)
'functional_1/Right_eye/conv1d_5/BiasAddМ
$functional_1/Right_eye/conv1d_5/ReluRelu0functional_1/Right_eye/conv1d_5/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
2&
$functional_1/Right_eye/conv1d_5/Relu
$functional_1/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$functional_1/concatenate/concat/axisЃ
functional_1/concatenate/concatConcatV21functional_1/Left_eye/conv1d_2/Relu:activations:02functional_1/Right_eye/conv1d_5/Relu:activations:0-functional_1/concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ
 2!
functional_1/concatenate/concat
functional_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
functional_1/flatten/ConstЩ
functional_1/flatten/ReshapeReshape(functional_1/concatenate/concat:output:0#functional_1/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
functional_1/flatten/ReshapeШ
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp1functional_1_dense_matmul_readvariableop_resource* 
_output_shapes
:
Р*
dtype02*
(functional_1/dense/MatMul/ReadVariableOpД
functional_1/dense/MatMul/CastCast0functional_1/dense/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:
Р2 
functional_1/dense/MatMul/CastО
functional_1/dense/MatMulMatMul%functional_1/flatten/Reshape:output:0"functional_1/dense/MatMul/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_1/dense/MatMulЦ
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp2functional_1_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOpВ
functional_1/dense/BiasAdd/CastCast1functional_1/dense/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:2!
functional_1/dense/BiasAdd/CastР
functional_1/dense/BiasAddBiasAdd#functional_1/dense/MatMul:product:0#functional_1/dense/BiasAdd/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_1/dense/BiasAdd
functional_1/dense/ReluRelu#functional_1/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_1/dense/ReluЮ
*functional_1/dense_1/MatMul/ReadVariableOpReadVariableOp3functional_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02,
*functional_1/dense_1/MatMul/ReadVariableOpК
 functional_1/dense_1/MatMul/CastCast2functional_1/dense_1/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:
2"
 functional_1/dense_1/MatMul/CastФ
functional_1/dense_1/MatMulMatMul%functional_1/dense/Relu:activations:0$functional_1/dense_1/MatMul/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_1/dense_1/MatMulЬ
+functional_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+functional_1/dense_1/BiasAdd/ReadVariableOpИ
!functional_1/dense_1/BiasAdd/CastCast3functional_1/dense_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:2#
!functional_1/dense_1/BiasAdd/CastШ
functional_1/dense_1/BiasAddBiasAdd%functional_1/dense_1/MatMul:product:0%functional_1/dense_1/BiasAdd/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_1/dense_1/BiasAdd
functional_1/dense_1/ReluRelu%functional_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_1/dense_1/ReluЭ
*functional_1/dense_2/MatMul/ReadVariableOpReadVariableOp3functional_1_dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02,
*functional_1/dense_2/MatMul/ReadVariableOpЙ
 functional_1/dense_2/MatMul/CastCast2functional_1/dense_2/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	@2"
 functional_1/dense_2/MatMul/CastХ
functional_1/dense_2/MatMulMatMul'functional_1/dense_1/Relu:activations:0$functional_1/dense_2/MatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
functional_1/dense_2/MatMulЫ
+functional_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+functional_1/dense_2/BiasAdd/ReadVariableOpЗ
!functional_1/dense_2/BiasAdd/CastCast3functional_1/dense_2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2#
!functional_1/dense_2/BiasAdd/CastЧ
functional_1/dense_2/BiasAddBiasAdd%functional_1/dense_2/MatMul:product:0%functional_1/dense_2/BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
functional_1/dense_2/BiasAdd
functional_1/dense_2/ReluRelu%functional_1/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
functional_1/dense_2/ReluЬ
*functional_1/dense_3/MatMul/ReadVariableOpReadVariableOp3functional_1_dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*functional_1/dense_3/MatMul/ReadVariableOpИ
 functional_1/dense_3/MatMul/CastCast2functional_1/dense_3/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes

:@2"
 functional_1/dense_3/MatMul/CastХ
functional_1/dense_3/MatMulMatMul'functional_1/dense_2/Relu:activations:0$functional_1/dense_3/MatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
functional_1/dense_3/MatMulЫ
+functional_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+functional_1/dense_3/BiasAdd/ReadVariableOpЗ
!functional_1/dense_3/BiasAdd/CastCast3functional_1/dense_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2#
!functional_1/dense_3/BiasAdd/CastЧ
functional_1/dense_3/BiasAddBiasAdd%functional_1/dense_3/MatMul:product:0%functional_1/dense_3/BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
functional_1/dense_3/BiasAdd
functional_1/CastCast%functional_1/dense_3/BiasAdd:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2
functional_1/Casti
IdentityIdentityfunctional_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd:џџџџџџџџџd:::::::::::::::::::::U Q
/
_output_shapes
:џџџџџџџџџd

_user_specified_nameLeft:VR
/
_output_shapes
:џџџџџџџџџd

_user_specified_nameRight
УЈ
н&
$__inference__traced_restore_33198770
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias%
!assignvariableop_6_dense_3_kernel#
assignvariableop_7_dense_3_bias'
#assignvariableop_8_cond_1_adam_iter)
%assignvariableop_9_cond_1_adam_beta_1*
&assignvariableop_10_cond_1_adam_beta_2)
%assignvariableop_11_cond_1_adam_decay1
-assignvariableop_12_cond_1_adam_learning_rate%
!assignvariableop_13_conv1d_kernel#
assignvariableop_14_conv1d_bias'
#assignvariableop_15_conv1d_1_kernel%
!assignvariableop_16_conv1d_1_bias'
#assignvariableop_17_conv1d_2_kernel%
!assignvariableop_18_conv1d_2_bias'
#assignvariableop_19_conv1d_3_kernel%
!assignvariableop_20_conv1d_3_bias'
#assignvariableop_21_conv1d_4_kernel%
!assignvariableop_22_conv1d_4_bias'
#assignvariableop_23_conv1d_5_kernel%
!assignvariableop_24_conv1d_5_bias*
&assignvariableop_25_current_loss_scale"
assignvariableop_26_good_steps
assignvariableop_27_total
assignvariableop_28_count
assignvariableop_29_total_1
assignvariableop_30_count_12
.assignvariableop_31_cond_1_adam_dense_kernel_m0
,assignvariableop_32_cond_1_adam_dense_bias_m4
0assignvariableop_33_cond_1_adam_dense_1_kernel_m2
.assignvariableop_34_cond_1_adam_dense_1_bias_m4
0assignvariableop_35_cond_1_adam_dense_2_kernel_m2
.assignvariableop_36_cond_1_adam_dense_2_bias_m4
0assignvariableop_37_cond_1_adam_dense_3_kernel_m2
.assignvariableop_38_cond_1_adam_dense_3_bias_m3
/assignvariableop_39_cond_1_adam_conv1d_kernel_m1
-assignvariableop_40_cond_1_adam_conv1d_bias_m5
1assignvariableop_41_cond_1_adam_conv1d_1_kernel_m3
/assignvariableop_42_cond_1_adam_conv1d_1_bias_m5
1assignvariableop_43_cond_1_adam_conv1d_2_kernel_m3
/assignvariableop_44_cond_1_adam_conv1d_2_bias_m5
1assignvariableop_45_cond_1_adam_conv1d_3_kernel_m3
/assignvariableop_46_cond_1_adam_conv1d_3_bias_m5
1assignvariableop_47_cond_1_adam_conv1d_4_kernel_m3
/assignvariableop_48_cond_1_adam_conv1d_4_bias_m5
1assignvariableop_49_cond_1_adam_conv1d_5_kernel_m3
/assignvariableop_50_cond_1_adam_conv1d_5_bias_m2
.assignvariableop_51_cond_1_adam_dense_kernel_v0
,assignvariableop_52_cond_1_adam_dense_bias_v4
0assignvariableop_53_cond_1_adam_dense_1_kernel_v2
.assignvariableop_54_cond_1_adam_dense_1_bias_v4
0assignvariableop_55_cond_1_adam_dense_2_kernel_v2
.assignvariableop_56_cond_1_adam_dense_2_bias_v4
0assignvariableop_57_cond_1_adam_dense_3_kernel_v2
.assignvariableop_58_cond_1_adam_dense_3_bias_v3
/assignvariableop_59_cond_1_adam_conv1d_kernel_v1
-assignvariableop_60_cond_1_adam_conv1d_bias_v5
1assignvariableop_61_cond_1_adam_conv1d_1_kernel_v3
/assignvariableop_62_cond_1_adam_conv1d_1_bias_v5
1assignvariableop_63_cond_1_adam_conv1d_2_kernel_v3
/assignvariableop_64_cond_1_adam_conv1d_2_bias_v5
1assignvariableop_65_cond_1_adam_conv1d_3_kernel_v3
/assignvariableop_66_cond_1_adam_conv1d_3_bias_v5
1assignvariableop_67_cond_1_adam_conv1d_4_kernel_v3
/assignvariableop_68_cond_1_adam_conv1d_4_bias_v5
1assignvariableop_69_cond_1_adam_conv1d_5_kernel_v3
/assignvariableop_70_cond_1_adam_conv1d_5_bias_v
identity_72ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_70ЂAssignVariableOp_8ЂAssignVariableOp_9$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*#
value#B#HB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEBBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЁ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*Ѕ
valueBHB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ж
_output_shapesЃ
 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*V
dtypesL
J2H		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ђ
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2І
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Є
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4І
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Є
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6І
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Є
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8Ј
AssignVariableOp_8AssignVariableOp#assignvariableop_8_cond_1_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Њ
AssignVariableOp_9AssignVariableOp%assignvariableop_9_cond_1_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ў
AssignVariableOp_10AssignVariableOp&assignvariableop_10_cond_1_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11­
AssignVariableOp_11AssignVariableOp%assignvariableop_11_cond_1_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Е
AssignVariableOp_12AssignVariableOp-assignvariableop_12_cond_1_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Љ
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv1d_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ї
AssignVariableOp_14AssignVariableOpassignvariableop_14_conv1d_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ћ
AssignVariableOp_15AssignVariableOp#assignvariableop_15_conv1d_1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Љ
AssignVariableOp_16AssignVariableOp!assignvariableop_16_conv1d_1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ћ
AssignVariableOp_17AssignVariableOp#assignvariableop_17_conv1d_2_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Љ
AssignVariableOp_18AssignVariableOp!assignvariableop_18_conv1d_2_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ћ
AssignVariableOp_19AssignVariableOp#assignvariableop_19_conv1d_3_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Љ
AssignVariableOp_20AssignVariableOp!assignvariableop_20_conv1d_3_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ћ
AssignVariableOp_21AssignVariableOp#assignvariableop_21_conv1d_4_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Љ
AssignVariableOp_22AssignVariableOp!assignvariableop_22_conv1d_4_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ћ
AssignVariableOp_23AssignVariableOp#assignvariableop_23_conv1d_5_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Љ
AssignVariableOp_24AssignVariableOp!assignvariableop_24_conv1d_5_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ў
AssignVariableOp_25AssignVariableOp&assignvariableop_25_current_loss_scaleIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_26І
AssignVariableOp_26AssignVariableOpassignvariableop_26_good_stepsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ё
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ё
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ѓ
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ѓ
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ж
AssignVariableOp_31AssignVariableOp.assignvariableop_31_cond_1_adam_dense_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Д
AssignVariableOp_32AssignVariableOp,assignvariableop_32_cond_1_adam_dense_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33И
AssignVariableOp_33AssignVariableOp0assignvariableop_33_cond_1_adam_dense_1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ж
AssignVariableOp_34AssignVariableOp.assignvariableop_34_cond_1_adam_dense_1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35И
AssignVariableOp_35AssignVariableOp0assignvariableop_35_cond_1_adam_dense_2_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ж
AssignVariableOp_36AssignVariableOp.assignvariableop_36_cond_1_adam_dense_2_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37И
AssignVariableOp_37AssignVariableOp0assignvariableop_37_cond_1_adam_dense_3_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ж
AssignVariableOp_38AssignVariableOp.assignvariableop_38_cond_1_adam_dense_3_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39З
AssignVariableOp_39AssignVariableOp/assignvariableop_39_cond_1_adam_conv1d_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Е
AssignVariableOp_40AssignVariableOp-assignvariableop_40_cond_1_adam_conv1d_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Й
AssignVariableOp_41AssignVariableOp1assignvariableop_41_cond_1_adam_conv1d_1_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42З
AssignVariableOp_42AssignVariableOp/assignvariableop_42_cond_1_adam_conv1d_1_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Й
AssignVariableOp_43AssignVariableOp1assignvariableop_43_cond_1_adam_conv1d_2_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44З
AssignVariableOp_44AssignVariableOp/assignvariableop_44_cond_1_adam_conv1d_2_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Й
AssignVariableOp_45AssignVariableOp1assignvariableop_45_cond_1_adam_conv1d_3_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46З
AssignVariableOp_46AssignVariableOp/assignvariableop_46_cond_1_adam_conv1d_3_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Й
AssignVariableOp_47AssignVariableOp1assignvariableop_47_cond_1_adam_conv1d_4_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48З
AssignVariableOp_48AssignVariableOp/assignvariableop_48_cond_1_adam_conv1d_4_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Й
AssignVariableOp_49AssignVariableOp1assignvariableop_49_cond_1_adam_conv1d_5_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50З
AssignVariableOp_50AssignVariableOp/assignvariableop_50_cond_1_adam_conv1d_5_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Ж
AssignVariableOp_51AssignVariableOp.assignvariableop_51_cond_1_adam_dense_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Д
AssignVariableOp_52AssignVariableOp,assignvariableop_52_cond_1_adam_dense_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53И
AssignVariableOp_53AssignVariableOp0assignvariableop_53_cond_1_adam_dense_1_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Ж
AssignVariableOp_54AssignVariableOp.assignvariableop_54_cond_1_adam_dense_1_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55И
AssignVariableOp_55AssignVariableOp0assignvariableop_55_cond_1_adam_dense_2_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Ж
AssignVariableOp_56AssignVariableOp.assignvariableop_56_cond_1_adam_dense_2_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57И
AssignVariableOp_57AssignVariableOp0assignvariableop_57_cond_1_adam_dense_3_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58Ж
AssignVariableOp_58AssignVariableOp.assignvariableop_58_cond_1_adam_dense_3_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59З
AssignVariableOp_59AssignVariableOp/assignvariableop_59_cond_1_adam_conv1d_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Е
AssignVariableOp_60AssignVariableOp-assignvariableop_60_cond_1_adam_conv1d_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Й
AssignVariableOp_61AssignVariableOp1assignvariableop_61_cond_1_adam_conv1d_1_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62З
AssignVariableOp_62AssignVariableOp/assignvariableop_62_cond_1_adam_conv1d_1_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Й
AssignVariableOp_63AssignVariableOp1assignvariableop_63_cond_1_adam_conv1d_2_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64З
AssignVariableOp_64AssignVariableOp/assignvariableop_64_cond_1_adam_conv1d_2_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65Й
AssignVariableOp_65AssignVariableOp1assignvariableop_65_cond_1_adam_conv1d_3_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66З
AssignVariableOp_66AssignVariableOp/assignvariableop_66_cond_1_adam_conv1d_3_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Й
AssignVariableOp_67AssignVariableOp1assignvariableop_67_cond_1_adam_conv1d_4_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68З
AssignVariableOp_68AssignVariableOp/assignvariableop_68_cond_1_adam_conv1d_4_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69Й
AssignVariableOp_69AssignVariableOp1assignvariableop_69_cond_1_adam_conv1d_5_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70З
AssignVariableOp_70AssignVariableOp/assignvariableop_70_cond_1_adam_conv1d_5_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_709
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpј
Identity_71Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_71ы
Identity_72IdentityIdentity_71:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_72"#
identity_72Identity_72:output:0*Г
_input_shapesЁ
: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
І	
­
E__inference_dense_3_layer_call_and_return_conditional_losses_33198094

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpy
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes

:@2
MatMul/Caste
MatMulMatMulinputsMatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
BiasAdd/Casts
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ю
a
E__inference_reshape_layer_call_and_return_conditional_losses_33196302

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd:W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
х

*__inference_dense_2_layer_call_fn_33198082

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_331969382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Н
+__inference_Left_eye_layer_call_fn_33196473
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Left_eye_layer_call_and_return_conditional_losses_331964582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџd
!
_user_specified_name	input_1
 
Л
F__inference_conv1d_4_layer_call_and_return_conditional_losses_33196589

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpЄ
conv1d/ExpandDims_1/CastCast*conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@ 2
conv1d/ExpandDims_1/Castt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЉ
conv1d/ExpandDims_1
ExpandDimsconv1d/ExpandDims_1/Cast:y:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
BiasAdd/Cast~
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ/@:::S O
+
_output_shapes
:џџџџџџџџџ/@
 
_user_specified_nameinputs
ё
~
)__inference_conv1d_layer_call_fn_33198157

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv1d_layer_call_and_return_conditional_losses_331963282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ/@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџd	::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџd	
 
_user_specified_nameinputs
г
Є
/__inference_functional_1_layer_call_fn_33197153
left	
right
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallleftrightunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_331971102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd:џџџџџџџџџd::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
/
_output_shapes
:џџџџџџџџџd

_user_specified_nameLeft:VR
/
_output_shapes
:џџџџџџџџџd

_user_specified_nameRight


­
E__inference_dense_1_layer_call_and_return_conditional_losses_33196909

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOp{
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:
2
MatMul/Castf
MatMulMatMulinputsMatMul/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:2
BiasAdd/Castt
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

О
,__inference_Right_eye_layer_call_fn_33196738
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_Right_eye_layer_call_and_return_conditional_losses_331967232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџd
!
_user_specified_name	input_2

к
G__inference_Right_eye_layer_call_and_return_conditional_losses_33196661
input_2
conv1d_3_33196645
conv1d_3_33196647
conv1d_4_33196650
conv1d_4_33196652
conv1d_5_33196655
conv1d_5_33196657
identityЂ conv1d_3/StatefulPartitionedCallЂ conv1d_4/StatefulPartitionedCallЂ conv1d_5/StatefulPartitionedCallf
CastCastinput_2*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
Castф
reshape_1/PartitionedCallPartitionedCallCast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_331965292
reshape_1/PartitionedCallП
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv1d_3_33196645conv1d_3_33196647*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_3_layer_call_and_return_conditional_losses_331965552"
 conv1d_3/StatefulPartitionedCallЦ
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0conv1d_4_33196650conv1d_4_33196652*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_4_layer_call_and_return_conditional_losses_331965892"
 conv1d_4/StatefulPartitionedCallЦ
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0conv1d_5_33196655conv1d_5_33196657*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_5_layer_call_and_return_conditional_losses_331966232"
 conv1d_5/StatefulPartitionedCallъ
IdentityIdentity)conv1d_5/StatefulPartitionedCall:output:0!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd::::::2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџd
!
_user_specified_name	input_2
жA

F__inference_Left_eye_layer_call_and_return_conditional_losses_33197812

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource
identitye
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
CastV
reshape/ShapeShapeCast:y:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape/Reshape/shape/2Ш
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape
reshape/ReshapeReshapeCast:y:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2
reshape/Reshape
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/conv1d/ExpandDims/dimН
conv1d/conv1d/ExpandDims
ExpandDimsreshape/Reshape:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd	2
conv1d/conv1d/ExpandDimsЭ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOpЙ
conv1d/conv1d/ExpandDims_1/CastCast1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:	@2!
conv1d/conv1d/ExpandDims_1/Cast
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimХ
conv1d/conv1d/ExpandDims_1
ExpandDims#conv1d/conv1d/ExpandDims_1/Cast:y:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d/conv1d/ExpandDims_1г
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@*
paddingVALID*
strides
2
conv1d/conv1dЇ
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@*
squeeze_dims

§џџџџџџџџ2
conv1d/conv1d/SqueezeЁ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv1d/BiasAdd/ReadVariableOp
conv1d/BiasAdd/CastCast%conv1d/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
conv1d/BiasAdd/Cast
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0conv1d/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
conv1d/BiasAddq
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
conv1d/Relu
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
conv1d_1/conv1d/ExpandDims/dimФ
conv1d_1/conv1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@2
conv1d_1/conv1d/ExpandDimsг
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpП
!conv1d_1/conv1d/ExpandDims_1/CastCast3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@ 2#
!conv1d_1/conv1d/ExpandDims_1/Cast
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dimЭ
conv1d_1/conv1d/ExpandDims_1
ExpandDims%conv1d_1/conv1d/ExpandDims_1/Cast:y:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d_1/conv1d/ExpandDims_1л
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv1d_1/conv1d­
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџ2
conv1d_1/conv1d/SqueezeЇ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_1/BiasAdd/ReadVariableOp
conv1d_1/BiasAdd/CastCast'conv1d_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
conv1d_1/BiasAdd/CastЂ
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0conv1d_1/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
conv1d_1/Relu
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
conv1d_2/conv1d/ExpandDims/dimЦ
conv1d_2/conv1d/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv1d_2/conv1d/ExpandDimsг
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpП
!conv1d_2/conv1d/ExpandDims_1/CastCast3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
: 2#
!conv1d_2/conv1d/ExpandDims_1/Cast
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dimЭ
conv1d_2/conv1d/ExpandDims_1
ExpandDims%conv1d_2/conv1d/ExpandDims_1/Cast:y:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_2/conv1d/ExpandDims_1л
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
*
paddingVALID*
strides
2
conv1d_2/conv1d­
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
*
squeeze_dims

§џџџџџџџџ2
conv1d_2/conv1d/SqueezeЇ
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp
conv1d_2/BiasAdd/CastCast'conv1d_2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
conv1d_2/BiasAdd/CastЂ
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0conv1d_2/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
conv1d_2/BiasAddw
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
conv1d_2/Relus
IdentityIdentityconv1d_2/Relu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd:::::::W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
§
й
G__inference_Right_eye_layer_call_and_return_conditional_losses_33196723

inputs
conv1d_3_33196707
conv1d_3_33196709
conv1d_4_33196712
conv1d_4_33196714
conv1d_5_33196717
conv1d_5_33196719
identityЂ conv1d_3/StatefulPartitionedCallЂ conv1d_4/StatefulPartitionedCallЂ conv1d_5/StatefulPartitionedCalle
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
Castф
reshape_1/PartitionedCallPartitionedCallCast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_331965292
reshape_1/PartitionedCallП
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv1d_3_33196707conv1d_3_33196709*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_3_layer_call_and_return_conditional_losses_331965552"
 conv1d_3/StatefulPartitionedCallЦ
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0conv1d_4_33196712conv1d_4_33196714*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_4_layer_call_and_return_conditional_losses_331965892"
 conv1d_4/StatefulPartitionedCallЦ
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0conv1d_5_33196717conv1d_5_33196719*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_5_layer_call_and_return_conditional_losses_331966232"
 conv1d_5/StatefulPartitionedCallъ
IdentityIdentity)conv1d_5/StatefulPartitionedCall:output:0!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd::::::2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs

к
G__inference_Right_eye_layer_call_and_return_conditional_losses_33196640
input_2
conv1d_3_33196566
conv1d_3_33196568
conv1d_4_33196600
conv1d_4_33196602
conv1d_5_33196634
conv1d_5_33196636
identityЂ conv1d_3/StatefulPartitionedCallЂ conv1d_4/StatefulPartitionedCallЂ conv1d_5/StatefulPartitionedCallf
CastCastinput_2*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
Castф
reshape_1/PartitionedCallPartitionedCallCast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_331965292
reshape_1/PartitionedCallП
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv1d_3_33196566conv1d_3_33196568*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_3_layer_call_and_return_conditional_losses_331965552"
 conv1d_3/StatefulPartitionedCallЦ
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0conv1d_4_33196600conv1d_4_33196602*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_4_layer_call_and_return_conditional_losses_331965892"
 conv1d_4/StatefulPartitionedCallЦ
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0conv1d_5_33196634conv1d_5_33196636*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_5_layer_call_and_return_conditional_losses_331966232"
 conv1d_5/StatefulPartitionedCallъ
IdentityIdentity)conv1d_5/StatefulPartitionedCall:output:0!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd::::::2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџd
!
_user_specified_name	input_2
 
Л
F__inference_conv1d_3_layer_call_and_return_conditional_losses_33198247

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd	2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpЄ
conv1d/ExpandDims_1/CastCast*conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:	@2
conv1d/ExpandDims_1/Castt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЉ
conv1d/ExpandDims_1
ExpandDimsconv1d/ExpandDims_1/Cast:y:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
BiasAdd/Cast~
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџd	:::S O
+
_output_shapes
:џџџџџџџџџd	
 
_user_specified_nameinputs
і

+__inference_conv1d_1_layer_call_fn_33198184

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_1_layer_call_and_return_conditional_losses_331963622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ/@::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ/@
 
_user_specified_nameinputs
п
г
F__inference_Left_eye_layer_call_and_return_conditional_losses_33196413
input_1
conv1d_33196339
conv1d_33196341
conv1d_1_33196373
conv1d_1_33196375
conv1d_2_33196407
conv1d_2_33196409
identityЂconv1d/StatefulPartitionedCallЂ conv1d_1/StatefulPartitionedCallЂ conv1d_2/StatefulPartitionedCallf
CastCastinput_1*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
Castо
reshape/PartitionedCallPartitionedCallCast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_331963022
reshape/PartitionedCallГ
conv1d/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1d_33196339conv1d_33196341*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv1d_layer_call_and_return_conditional_losses_331963282 
conv1d/StatefulPartitionedCallФ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_33196373conv1d_1_33196375*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_1_layer_call_and_return_conditional_losses_331963622"
 conv1d_1/StatefulPartitionedCallЦ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_2_33196407conv1d_2_33196409*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_331963962"
 conv1d_2/StatefulPartitionedCallш
IdentityIdentity)conv1d_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџd
!
_user_specified_name	input_1
і

+__inference_conv1d_3_layer_call_fn_33198256

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_3_layer_call_and_return_conditional_losses_331965552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ/@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџd	::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџd	
 
_user_specified_nameinputs
г
Є
/__inference_functional_1_layer_call_fn_33197254
left	
right
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallleftrightunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_331972112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd:џџџџџџџџџd::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
/
_output_shapes
:џџџџџџџџџd

_user_specified_nameLeft:VR
/
_output_shapes
:џџџџџџџџџd

_user_specified_nameRight
 
Л
F__inference_conv1d_2_layer_call_and_return_conditional_losses_33196396

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpЄ
conv1d/ExpandDims_1/CastCast*conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
: 2
conv1d/ExpandDims_1/Castt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЉ
conv1d/ExpandDims_1
ExpandDimsconv1d/ExpandDims_1/Cast:y:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
BiasAdd/Cast~
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ :::S O
+
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ч

*__inference_dense_1_layer_call_fn_33198060

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_331969092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


­
E__inference_dense_2_layer_call_and_return_conditional_losses_33196938

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOpz
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	@2
MatMul/Caste
MatMulMatMulinputsMatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
BiasAdd/Casts
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
І	
­
E__inference_dense_3_layer_call_and_return_conditional_losses_33196966

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOpy
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes

:@2
MatMul/Caste
MatMulMatMulinputsMatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
BiasAdd/Casts
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ@:::O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
м
в
F__inference_Left_eye_layer_call_and_return_conditional_losses_33196496

inputs
conv1d_33196480
conv1d_33196482
conv1d_1_33196485
conv1d_1_33196487
conv1d_2_33196490
conv1d_2_33196492
identityЂconv1d/StatefulPartitionedCallЂ conv1d_1/StatefulPartitionedCallЂ conv1d_2/StatefulPartitionedCalle
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
Castо
reshape/PartitionedCallPartitionedCallCast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_331963022
reshape/PartitionedCallГ
conv1d/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1d_33196480conv1d_33196482*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv1d_layer_call_and_return_conditional_losses_331963282 
conv1d/StatefulPartitionedCallФ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_33196485conv1d_1_33196487*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_1_layer_call_and_return_conditional_losses_331963622"
 conv1d_1/StatefulPartitionedCallЦ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_2_33196490conv1d_2_33196492*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_331963962"
 conv1d_2/StatefulPartitionedCallш
IdentityIdentity)conv1d_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Е
a
E__inference_flatten_layer_call_and_return_conditional_losses_33196859

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ
 :S O
+
_output_shapes
:џџџџџџџџџ
 
 
_user_specified_nameinputs

Н
,__inference_Right_eye_layer_call_fn_33197975

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_Right_eye_layer_call_and_return_conditional_losses_331966852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
№
c
G__inference_reshape_1_layer_call_and_return_conditional_losses_33196529

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd:W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs

я
!__inference__traced_save_33198547
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop/
+savev2_cond_1_adam_iter_read_readvariableop	1
-savev2_cond_1_adam_beta_1_read_readvariableop1
-savev2_cond_1_adam_beta_2_read_readvariableop0
,savev2_cond_1_adam_decay_read_readvariableop8
4savev2_cond_1_adam_learning_rate_read_readvariableop,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop.
*savev2_conv1d_4_kernel_read_readvariableop,
(savev2_conv1d_4_bias_read_readvariableop.
*savev2_conv1d_5_kernel_read_readvariableop,
(savev2_conv1d_5_bias_read_readvariableop1
-savev2_current_loss_scale_read_readvariableop)
%savev2_good_steps_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop9
5savev2_cond_1_adam_dense_kernel_m_read_readvariableop7
3savev2_cond_1_adam_dense_bias_m_read_readvariableop;
7savev2_cond_1_adam_dense_1_kernel_m_read_readvariableop9
5savev2_cond_1_adam_dense_1_bias_m_read_readvariableop;
7savev2_cond_1_adam_dense_2_kernel_m_read_readvariableop9
5savev2_cond_1_adam_dense_2_bias_m_read_readvariableop;
7savev2_cond_1_adam_dense_3_kernel_m_read_readvariableop9
5savev2_cond_1_adam_dense_3_bias_m_read_readvariableop:
6savev2_cond_1_adam_conv1d_kernel_m_read_readvariableop8
4savev2_cond_1_adam_conv1d_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv1d_1_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv1d_1_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv1d_2_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv1d_2_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv1d_3_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv1d_3_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv1d_4_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv1d_4_bias_m_read_readvariableop<
8savev2_cond_1_adam_conv1d_5_kernel_m_read_readvariableop:
6savev2_cond_1_adam_conv1d_5_bias_m_read_readvariableop9
5savev2_cond_1_adam_dense_kernel_v_read_readvariableop7
3savev2_cond_1_adam_dense_bias_v_read_readvariableop;
7savev2_cond_1_adam_dense_1_kernel_v_read_readvariableop9
5savev2_cond_1_adam_dense_1_bias_v_read_readvariableop;
7savev2_cond_1_adam_dense_2_kernel_v_read_readvariableop9
5savev2_cond_1_adam_dense_2_bias_v_read_readvariableop;
7savev2_cond_1_adam_dense_3_kernel_v_read_readvariableop9
5savev2_cond_1_adam_dense_3_bias_v_read_readvariableop:
6savev2_cond_1_adam_conv1d_kernel_v_read_readvariableop8
4savev2_cond_1_adam_conv1d_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv1d_1_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv1d_1_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv1d_2_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv1d_2_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv1d_3_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv1d_3_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv1d_4_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv1d_4_bias_v_read_readvariableop<
8savev2_cond_1_adam_conv1d_5_kernel_v_read_readvariableop:
6savev2_cond_1_adam_conv1d_5_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_cd90b134fbf14112a6b9dcdfd0966fb0/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*#
value#B#HB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEBBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*Ѕ
valueBHB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesц
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop+savev2_cond_1_adam_iter_read_readvariableop-savev2_cond_1_adam_beta_1_read_readvariableop-savev2_cond_1_adam_beta_2_read_readvariableop,savev2_cond_1_adam_decay_read_readvariableop4savev2_cond_1_adam_learning_rate_read_readvariableop(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop*savev2_conv1d_4_kernel_read_readvariableop(savev2_conv1d_4_bias_read_readvariableop*savev2_conv1d_5_kernel_read_readvariableop(savev2_conv1d_5_bias_read_readvariableop-savev2_current_loss_scale_read_readvariableop%savev2_good_steps_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_cond_1_adam_dense_kernel_m_read_readvariableop3savev2_cond_1_adam_dense_bias_m_read_readvariableop7savev2_cond_1_adam_dense_1_kernel_m_read_readvariableop5savev2_cond_1_adam_dense_1_bias_m_read_readvariableop7savev2_cond_1_adam_dense_2_kernel_m_read_readvariableop5savev2_cond_1_adam_dense_2_bias_m_read_readvariableop7savev2_cond_1_adam_dense_3_kernel_m_read_readvariableop5savev2_cond_1_adam_dense_3_bias_m_read_readvariableop6savev2_cond_1_adam_conv1d_kernel_m_read_readvariableop4savev2_cond_1_adam_conv1d_bias_m_read_readvariableop8savev2_cond_1_adam_conv1d_1_kernel_m_read_readvariableop6savev2_cond_1_adam_conv1d_1_bias_m_read_readvariableop8savev2_cond_1_adam_conv1d_2_kernel_m_read_readvariableop6savev2_cond_1_adam_conv1d_2_bias_m_read_readvariableop8savev2_cond_1_adam_conv1d_3_kernel_m_read_readvariableop6savev2_cond_1_adam_conv1d_3_bias_m_read_readvariableop8savev2_cond_1_adam_conv1d_4_kernel_m_read_readvariableop6savev2_cond_1_adam_conv1d_4_bias_m_read_readvariableop8savev2_cond_1_adam_conv1d_5_kernel_m_read_readvariableop6savev2_cond_1_adam_conv1d_5_bias_m_read_readvariableop5savev2_cond_1_adam_dense_kernel_v_read_readvariableop3savev2_cond_1_adam_dense_bias_v_read_readvariableop7savev2_cond_1_adam_dense_1_kernel_v_read_readvariableop5savev2_cond_1_adam_dense_1_bias_v_read_readvariableop7savev2_cond_1_adam_dense_2_kernel_v_read_readvariableop5savev2_cond_1_adam_dense_2_bias_v_read_readvariableop7savev2_cond_1_adam_dense_3_kernel_v_read_readvariableop5savev2_cond_1_adam_dense_3_bias_v_read_readvariableop6savev2_cond_1_adam_conv1d_kernel_v_read_readvariableop4savev2_cond_1_adam_conv1d_bias_v_read_readvariableop8savev2_cond_1_adam_conv1d_1_kernel_v_read_readvariableop6savev2_cond_1_adam_conv1d_1_bias_v_read_readvariableop8savev2_cond_1_adam_conv1d_2_kernel_v_read_readvariableop6savev2_cond_1_adam_conv1d_2_bias_v_read_readvariableop8savev2_cond_1_adam_conv1d_3_kernel_v_read_readvariableop6savev2_cond_1_adam_conv1d_3_bias_v_read_readvariableop8savev2_cond_1_adam_conv1d_4_kernel_v_read_readvariableop6savev2_cond_1_adam_conv1d_4_bias_v_read_readvariableop8savev2_cond_1_adam_conv1d_5_kernel_v_read_readvariableop6savev2_cond_1_adam_conv1d_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *V
dtypesL
J2H		2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ь
_input_shapesк
з: :
Р::
::	@:@:@:: : : : : :	@:@:@ : : ::	@:@:@ : : :: : : : : : :
Р::
::	@:@:@::	@:@:@ : : ::	@:@:@ : : ::
Р::
::	@:@:@::	@:@:@ : : ::	@:@:@ : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
Р:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:	@: 

_output_shapes
:@:($
"
_output_shapes
:@ : 

_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
::($
"
_output_shapes
:	@: 

_output_shapes
:@:($
"
_output_shapes
:@ : 

_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :& "
 
_output_shapes
:
Р:!!

_output_shapes	
::&""
 
_output_shapes
:
:!#

_output_shapes	
::%$!

_output_shapes
:	@: %

_output_shapes
:@:$& 

_output_shapes

:@: '

_output_shapes
::(($
"
_output_shapes
:	@: )

_output_shapes
:@:(*$
"
_output_shapes
:@ : +

_output_shapes
: :(,$
"
_output_shapes
: : -

_output_shapes
::(.$
"
_output_shapes
:	@: /

_output_shapes
:@:(0$
"
_output_shapes
:@ : 1

_output_shapes
: :(2$
"
_output_shapes
: : 3

_output_shapes
::&4"
 
_output_shapes
:
Р:!5

_output_shapes	
::&6"
 
_output_shapes
:
:!7

_output_shapes	
::%8!

_output_shapes
:	@: 9

_output_shapes
:@:$: 

_output_shapes

:@: ;

_output_shapes
::(<$
"
_output_shapes
:	@: =

_output_shapes
:@:(>$
"
_output_shapes
:@ : ?

_output_shapes
: :(@$
"
_output_shapes
: : A

_output_shapes
::(B$
"
_output_shapes
:	@: C

_output_shapes
:@:(D$
"
_output_shapes
:@ : E

_output_shapes
: :(F$
"
_output_shapes
: : G

_output_shapes
::H

_output_shapes
: 
нB

G__inference_Right_eye_layer_call_and_return_conditional_losses_33197902

inputs8
4conv1d_3_conv1d_expanddims_1_readvariableop_resource,
(conv1d_3_biasadd_readvariableop_resource8
4conv1d_4_conv1d_expanddims_1_readvariableop_resource,
(conv1d_4_biasadd_readvariableop_resource8
4conv1d_5_conv1d_expanddims_1_readvariableop_resource,
(conv1d_5_biasadd_readvariableop_resource
identitye
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
CastZ
reshape_1/ShapeShapeCast:y:0*
T0*
_output_shapes
:2
reshape_1/Shape
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape_1/Reshape/shape/2в
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape
reshape_1/ReshapeReshapeCast:y:0 reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2
reshape_1/Reshape
conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
conv1d_3/conv1d/ExpandDims/dimХ
conv1d_3/conv1d/ExpandDims
ExpandDimsreshape_1/Reshape:output:0'conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd	2
conv1d_3/conv1d/ExpandDimsг
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02-
+conv1d_3/conv1d/ExpandDims_1/ReadVariableOpП
!conv1d_3/conv1d/ExpandDims_1/CastCast3conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:	@2#
!conv1d_3/conv1d/ExpandDims_1/Cast
 conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_3/conv1d/ExpandDims_1/dimЭ
conv1d_3/conv1d/ExpandDims_1
ExpandDims%conv1d_3/conv1d/ExpandDims_1/Cast:y:0)conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d_3/conv1d/ExpandDims_1л
conv1d_3/conv1dConv2D#conv1d_3/conv1d/ExpandDims:output:0%conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@*
paddingVALID*
strides
2
conv1d_3/conv1d­
conv1d_3/conv1d/SqueezeSqueezeconv1d_3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@*
squeeze_dims

§џџџџџџџџ2
conv1d_3/conv1d/SqueezeЇ
conv1d_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_3/BiasAdd/ReadVariableOp
conv1d_3/BiasAdd/CastCast'conv1d_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
conv1d_3/BiasAdd/CastЂ
conv1d_3/BiasAddBiasAdd conv1d_3/conv1d/Squeeze:output:0conv1d_3/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
conv1d_3/BiasAddw
conv1d_3/ReluReluconv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
conv1d_3/Relu
conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
conv1d_4/conv1d/ExpandDims/dimЦ
conv1d_4/conv1d/ExpandDims
ExpandDimsconv1d_3/Relu:activations:0'conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@2
conv1d_4/conv1d/ExpandDimsг
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02-
+conv1d_4/conv1d/ExpandDims_1/ReadVariableOpП
!conv1d_4/conv1d/ExpandDims_1/CastCast3conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@ 2#
!conv1d_4/conv1d/ExpandDims_1/Cast
 conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_4/conv1d/ExpandDims_1/dimЭ
conv1d_4/conv1d/ExpandDims_1
ExpandDims%conv1d_4/conv1d/ExpandDims_1/Cast:y:0)conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d_4/conv1d/ExpandDims_1л
conv1d_4/conv1dConv2D#conv1d_4/conv1d/ExpandDims:output:0%conv1d_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv1d_4/conv1d­
conv1d_4/conv1d/SqueezeSqueezeconv1d_4/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџ2
conv1d_4/conv1d/SqueezeЇ
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_4/BiasAdd/ReadVariableOp
conv1d_4/BiasAdd/CastCast'conv1d_4/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
conv1d_4/BiasAdd/CastЂ
conv1d_4/BiasAddBiasAdd conv1d_4/conv1d/Squeeze:output:0conv1d_4/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
conv1d_4/BiasAddw
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
conv1d_4/Relu
conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
conv1d_5/conv1d/ExpandDims/dimЦ
conv1d_5/conv1d/ExpandDims
ExpandDimsconv1d_4/Relu:activations:0'conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv1d_5/conv1d/ExpandDimsг
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_5/conv1d/ExpandDims_1/ReadVariableOpП
!conv1d_5/conv1d/ExpandDims_1/CastCast3conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
: 2#
!conv1d_5/conv1d/ExpandDims_1/Cast
 conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_5/conv1d/ExpandDims_1/dimЭ
conv1d_5/conv1d/ExpandDims_1
ExpandDims%conv1d_5/conv1d/ExpandDims_1/Cast:y:0)conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_5/conv1d/ExpandDims_1л
conv1d_5/conv1dConv2D#conv1d_5/conv1d/ExpandDims:output:0%conv1d_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
*
paddingVALID*
strides
2
conv1d_5/conv1d­
conv1d_5/conv1d/SqueezeSqueezeconv1d_5/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
*
squeeze_dims

§џџџџџџџџ2
conv1d_5/conv1d/SqueezeЇ
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_5/BiasAdd/ReadVariableOp
conv1d_5/BiasAdd/CastCast'conv1d_5/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
conv1d_5/BiasAdd/CastЂ
conv1d_5/BiasAddBiasAdd conv1d_5/conv1d/Squeeze:output:0conv1d_5/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
conv1d_5/BiasAddw
conv1d_5/ReluReluconv1d_5/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
conv1d_5/Relus
IdentityIdentityconv1d_5/Relu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd:::::::W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
к
d
H__inference_activation_layer_call_and_return_conditional_losses_33196987

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
 
Л
F__inference_conv1d_2_layer_call_and_return_conditional_losses_33198202

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpЄ
conv1d/ExpandDims_1/CastCast*conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
: 2
conv1d/ExpandDims_1/Castt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЉ
conv1d/ExpandDims_1
ExpandDimsconv1d/ExpandDims_1/Cast:y:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
BiasAdd/Cast~
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ :::S O
+
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

Й
D__inference_conv1d_layer_call_and_return_conditional_losses_33196328

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd	2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpЄ
conv1d/ExpandDims_1/CastCast*conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:	@2
conv1d/ExpandDims_1/Castt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЉ
conv1d/ExpandDims_1
ExpandDimsconv1d/ExpandDims_1/Cast:y:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
BiasAdd/Cast~
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџd	:::S O
+
_output_shapes
:џџџџџџџџџd	
 
_user_specified_nameinputs
Е
a
E__inference_flatten_layer_call_and_return_conditional_losses_33198011

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ
 :S O
+
_output_shapes
:џџџџџџџџџ
 
 
_user_specified_nameinputs
 
Л
F__inference_conv1d_5_layer_call_and_return_conditional_losses_33198301

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpЄ
conv1d/ExpandDims_1/CastCast*conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
: 2
conv1d/ExpandDims_1/Castt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЉ
conv1d/ExpandDims_1
ExpandDimsconv1d/ExpandDims_1/Cast:y:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
BiasAdd/Cast~
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ :::S O
+
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs


­
E__inference_dense_1_layer_call_and_return_conditional_losses_33198051

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOp{
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:
2
MatMul/Castf
MatMulMatMulinputsMatMul/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:2
BiasAdd/Castt
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ѓ
F
*__inference_flatten_layer_call_fn_33198016

inputs
identityЩ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_331968592
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2

Identity"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ
 :S O
+
_output_shapes
:џџџџџџџџџ
 
 
_user_specified_nameinputs

I
-__inference_activation_layer_call_fn_33198112

inputs
identityЫ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_331969872
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і

+__inference_conv1d_5_layer_call_fn_33198310

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_5_layer_call_and_return_conditional_losses_331966232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
§
й
G__inference_Right_eye_layer_call_and_return_conditional_losses_33196685

inputs
conv1d_3_33196669
conv1d_3_33196671
conv1d_4_33196674
conv1d_4_33196676
conv1d_5_33196679
conv1d_5_33196681
identityЂ conv1d_3/StatefulPartitionedCallЂ conv1d_4/StatefulPartitionedCallЂ conv1d_5/StatefulPartitionedCalle
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
Castф
reshape_1/PartitionedCallPartitionedCallCast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_331965292
reshape_1/PartitionedCallП
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv1d_3_33196669conv1d_3_33196671*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_3_layer_call_and_return_conditional_losses_331965552"
 conv1d_3/StatefulPartitionedCallЦ
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0conv1d_4_33196674conv1d_4_33196676*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_4_layer_call_and_return_conditional_losses_331965892"
 conv1d_4/StatefulPartitionedCallЦ
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0conv1d_5_33196679conv1d_5_33196681*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_5_layer_call_and_return_conditional_losses_331966232"
 conv1d_5/StatefulPartitionedCallъ
IdentityIdentity)conv1d_5/StatefulPartitionedCall:output:0!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd::::::2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
м
в
F__inference_Left_eye_layer_call_and_return_conditional_losses_33196458

inputs
conv1d_33196442
conv1d_33196444
conv1d_1_33196447
conv1d_1_33196449
conv1d_2_33196452
conv1d_2_33196454
identityЂconv1d/StatefulPartitionedCallЂ conv1d_1/StatefulPartitionedCallЂ conv1d_2/StatefulPartitionedCalle
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
Castо
reshape/PartitionedCallPartitionedCallCast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџd	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_331963022
reshape/PartitionedCallГ
conv1d/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1d_33196442conv1d_33196444*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ/@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv1d_layer_call_and_return_conditional_losses_331963282 
conv1d/StatefulPartitionedCallФ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_33196447conv1d_1_33196449*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_1_layer_call_and_return_conditional_losses_331963622"
 conv1d_1/StatefulPartitionedCallЦ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_2_33196452conv1d_2_33196454*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_331963962"
 conv1d_2/StatefulPartitionedCallш
IdentityIdentity)conv1d_2/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs


Ћ
C__inference_dense_layer_call_and_return_conditional_losses_33198029

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Р*
dtype02
MatMul/ReadVariableOp{
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:
Р2
MatMul/Castf
MatMulMatMulinputsMatMul/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:2
BiasAdd/Castt
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР:::P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
ў
М
+__inference_Left_eye_layer_call_fn_33197846

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Left_eye_layer_call_and_return_conditional_losses_331964962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
 
Л
F__inference_conv1d_1_layer_call_and_return_conditional_losses_33198175

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpЄ
conv1d/ExpandDims_1/CastCast*conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@ 2
conv1d/ExpandDims_1/Castt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЉ
conv1d/ExpandDims_1
ExpandDimsconv1d/ExpandDims_1/Cast:y:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
BiasAdd/Cast~
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ/@:::S O
+
_output_shapes
:џџџџџџџџџ/@
 
_user_specified_nameinputs

Й
D__inference_conv1d_layer_call_and_return_conditional_losses_33198148

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd	2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpЄ
conv1d/ExpandDims_1/CastCast*conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:	@2
conv1d/ExpandDims_1/Castt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЉ
conv1d/ExpandDims_1
ExpandDimsconv1d/ExpandDims_1/Cast:y:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpx
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
BiasAdd/Cast~
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџd	:::S O
+
_output_shapes
:џџџџџџџџџd	
 
_user_specified_nameinputs
ш
Ћ
/__inference_functional_1_layer_call_fn_33197700
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_331972112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd:џџџџџџџџџd::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
inputs/1

Н
,__inference_Right_eye_layer_call_fn_33197992

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_Right_eye_layer_call_and_return_conditional_losses_331967232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
у
}
(__inference_dense_layer_call_fn_33198038

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_331968802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџР::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
е.

J__inference_functional_1_layer_call_and_return_conditional_losses_33196996
left	
right
left_eye_33196777
left_eye_33196779
left_eye_33196781
left_eye_33196783
left_eye_33196785
left_eye_33196787
right_eye_33196824
right_eye_33196826
right_eye_33196828
right_eye_33196830
right_eye_33196832
right_eye_33196834
dense_33196891
dense_33196893
dense_1_33196920
dense_1_33196922
dense_2_33196949
dense_2_33196951
dense_3_33196977
dense_3_33196979
identityЂ Left_eye/StatefulPartitionedCallЂ!Right_eye/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallѕ
 Left_eye/StatefulPartitionedCallStatefulPartitionedCallleftleft_eye_33196777left_eye_33196779left_eye_33196781left_eye_33196783left_eye_33196785left_eye_33196787*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Left_eye_layer_call_and_return_conditional_losses_331964582"
 Left_eye/StatefulPartitionedCallџ
!Right_eye/StatefulPartitionedCallStatefulPartitionedCallrightright_eye_33196824right_eye_33196826right_eye_33196828right_eye_33196830right_eye_33196832right_eye_33196834*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_Right_eye_layer_call_and_return_conditional_losses_331966852#
!Right_eye/StatefulPartitionedCallИ
concatenate/PartitionedCallPartitionedCall)Left_eye/StatefulPartitionedCall:output:0*Right_eye/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_331968442
concatenate/PartitionedCallї
flatten/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_331968592
flatten/PartitionedCallЋ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_33196891dense_33196893*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_331968802
dense/StatefulPartitionedCallЛ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_33196920dense_1_33196922*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_331969092!
dense_1/StatefulPartitionedCallМ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_33196949dense_2_33196951*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_331969382!
dense_2/StatefulPartitionedCallМ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_33196977dense_3_33196979*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_331969662!
dense_3/StatefulPartitionedCall
CastCast(dense_3/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2
Castу
activation/PartitionedCallPartitionedCallCast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_331969872
activation/PartitionedCallФ
IdentityIdentity#activation/PartitionedCall:output:0!^Left_eye/StatefulPartitionedCall"^Right_eye/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd:џџџџџџџџџd::::::::::::::::::::2D
 Left_eye/StatefulPartitionedCall Left_eye/StatefulPartitionedCall2F
!Right_eye/StatefulPartitionedCall!Right_eye/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:U Q
/
_output_shapes
:џџџџџџџџџd

_user_specified_nameLeft:VR
/
_output_shapes
:џџџџџџџџџd

_user_specified_nameRight
і

+__inference_conv1d_2_layer_call_fn_33198211

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_conv1d_2_layer_call_and_return_conditional_losses_331963962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:џџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
к
d
H__inference_activation_layer_call_and_return_conditional_losses_33198107

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ю
a
E__inference_reshape_layer_call_and_return_conditional_losses_33198125

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape/shape/2 
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџd:W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
жA

F__inference_Left_eye_layer_call_and_return_conditional_losses_33197756

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource
identitye
CastCastinputs*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
CastV
reshape/ShapeShapeCast:y:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape/Reshape/shape/2Ш
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape
reshape/ReshapeReshapeCast:y:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2
reshape/Reshape
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/conv1d/ExpandDims/dimН
conv1d/conv1d/ExpandDims
ExpandDimsreshape/Reshape:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd	2
conv1d/conv1d/ExpandDimsЭ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOpЙ
conv1d/conv1d/ExpandDims_1/CastCast1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:	@2!
conv1d/conv1d/ExpandDims_1/Cast
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dimХ
conv1d/conv1d/ExpandDims_1
ExpandDims#conv1d/conv1d/ExpandDims_1/Cast:y:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2
conv1d/conv1d/ExpandDims_1г
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@*
paddingVALID*
strides
2
conv1d/conv1dЇ
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@*
squeeze_dims

§џџџџџџџџ2
conv1d/conv1d/SqueezeЁ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv1d/BiasAdd/ReadVariableOp
conv1d/BiasAdd/CastCast%conv1d/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
conv1d/BiasAdd/Cast
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0conv1d/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
conv1d/BiasAddq
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
conv1d/Relu
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
conv1d_1/conv1d/ExpandDims/dimФ
conv1d_1/conv1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@2
conv1d_1/conv1d/ExpandDimsг
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpП
!conv1d_1/conv1d/ExpandDims_1/CastCast3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@ 2#
!conv1d_1/conv1d/ExpandDims_1/Cast
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dimЭ
conv1d_1/conv1d/ExpandDims_1
ExpandDims%conv1d_1/conv1d/ExpandDims_1/Cast:y:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d_1/conv1d/ExpandDims_1л
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv1d_1/conv1d­
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџ2
conv1d_1/conv1d/SqueezeЇ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_1/BiasAdd/ReadVariableOp
conv1d_1/BiasAdd/CastCast'conv1d_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2
conv1d_1/BiasAdd/CastЂ
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0conv1d_1/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
conv1d_1/Relu
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
conv1d_2/conv1d/ExpandDims/dimЦ
conv1d_2/conv1d/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv1d_2/conv1d/ExpandDimsг
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpП
!conv1d_2/conv1d/ExpandDims_1/CastCast3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
: 2#
!conv1d_2/conv1d/ExpandDims_1/Cast
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dimЭ
conv1d_2/conv1d/ExpandDims_1
ExpandDims%conv1d_2/conv1d/ExpandDims_1/Cast:y:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_2/conv1d/ExpandDims_1л
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
*
paddingVALID*
strides
2
conv1d_2/conv1d­
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
*
squeeze_dims

§џџџџџџџџ2
conv1d_2/conv1d/SqueezeЇ
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp
conv1d_2/BiasAdd/CastCast'conv1d_2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
conv1d_2/BiasAdd/CastЂ
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0conv1d_2/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
conv1d_2/BiasAddw
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
conv1d_2/Relus
IdentityIdentityconv1d_2/Relu:activations:0*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd:::::::W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ю
s
I__inference_concatenate_layer_call_and_return_conditional_losses_33196844

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ
 2
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:џџџџџџџџџ
:џџџџџџџџџ
:S O
+
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs:SO
+
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
т.

J__inference_functional_1_layer_call_and_return_conditional_losses_33197211

inputs
inputs_1
left_eye_33197160
left_eye_33197162
left_eye_33197164
left_eye_33197166
left_eye_33197168
left_eye_33197170
right_eye_33197173
right_eye_33197175
right_eye_33197177
right_eye_33197179
right_eye_33197181
right_eye_33197183
dense_33197188
dense_33197190
dense_1_33197193
dense_1_33197195
dense_2_33197198
dense_2_33197200
dense_3_33197203
dense_3_33197205
identityЂ Left_eye/StatefulPartitionedCallЂ!Right_eye/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallї
 Left_eye/StatefulPartitionedCallStatefulPartitionedCallinputsleft_eye_33197160left_eye_33197162left_eye_33197164left_eye_33197166left_eye_33197168left_eye_33197170*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Left_eye_layer_call_and_return_conditional_losses_331964962"
 Left_eye/StatefulPartitionedCall
!Right_eye/StatefulPartitionedCallStatefulPartitionedCallinputs_1right_eye_33197173right_eye_33197175right_eye_33197177right_eye_33197179right_eye_33197181right_eye_33197183*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_Right_eye_layer_call_and_return_conditional_losses_331967232#
!Right_eye/StatefulPartitionedCallИ
concatenate/PartitionedCallPartitionedCall)Left_eye/StatefulPartitionedCall:output:0*Right_eye/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_331968442
concatenate/PartitionedCallї
flatten/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџР* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_331968592
flatten/PartitionedCallЋ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_33197188dense_33197190*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_331968802
dense/StatefulPartitionedCallЛ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_33197193dense_1_33197195*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_331969092!
dense_1/StatefulPartitionedCallМ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_33197198dense_2_33197200*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_331969382!
dense_2/StatefulPartitionedCallМ
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_33197203dense_3_33197205*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_331969662!
dense_3/StatefulPartitionedCall
CastCast(dense_3/StatefulPartitionedCall:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2
Castу
activation/PartitionedCallPartitionedCallCast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_331969872
activation/PartitionedCallФ
IdentityIdentity#activation/PartitionedCall:output:0!^Left_eye/StatefulPartitionedCall"^Right_eye/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd:џџџџџџџџџd::::::::::::::::::::2D
 Left_eye/StatefulPartitionedCall Left_eye/StatefulPartitionedCall2F
!Right_eye/StatefulPartitionedCall!Right_eye/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs:WS
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
ў
М
+__inference_Left_eye_layer_call_fn_33197829

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Left_eye_layer_call_and_return_conditional_losses_331964582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
фР
З	
J__inference_functional_1_layer_call_and_return_conditional_losses_33197459
inputs_0
inputs_1?
;left_eye_conv1d_conv1d_expanddims_1_readvariableop_resource3
/left_eye_conv1d_biasadd_readvariableop_resourceA
=left_eye_conv1d_1_conv1d_expanddims_1_readvariableop_resource5
1left_eye_conv1d_1_biasadd_readvariableop_resourceA
=left_eye_conv1d_2_conv1d_expanddims_1_readvariableop_resource5
1left_eye_conv1d_2_biasadd_readvariableop_resourceB
>right_eye_conv1d_3_conv1d_expanddims_1_readvariableop_resource6
2right_eye_conv1d_3_biasadd_readvariableop_resourceB
>right_eye_conv1d_4_conv1d_expanddims_1_readvariableop_resource6
2right_eye_conv1d_4_biasadd_readvariableop_resourceB
>right_eye_conv1d_5_conv1d_expanddims_1_readvariableop_resource6
2right_eye_conv1d_5_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identityy
Left_eye/CastCastinputs_0*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
Left_eye/Castq
Left_eye/reshape/ShapeShapeLeft_eye/Cast:y:0*
T0*
_output_shapes
:2
Left_eye/reshape/Shape
$Left_eye/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Left_eye/reshape/strided_slice/stack
&Left_eye/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Left_eye/reshape/strided_slice/stack_1
&Left_eye/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Left_eye/reshape/strided_slice/stack_2Ш
Left_eye/reshape/strided_sliceStridedSliceLeft_eye/reshape/Shape:output:0-Left_eye/reshape/strided_slice/stack:output:0/Left_eye/reshape/strided_slice/stack_1:output:0/Left_eye/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Left_eye/reshape/strided_slice
 Left_eye/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2"
 Left_eye/reshape/Reshape/shape/1
 Left_eye/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2"
 Left_eye/reshape/Reshape/shape/2ѕ
Left_eye/reshape/Reshape/shapePack'Left_eye/reshape/strided_slice:output:0)Left_eye/reshape/Reshape/shape/1:output:0)Left_eye/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2 
Left_eye/reshape/Reshape/shapeБ
Left_eye/reshape/ReshapeReshapeLeft_eye/Cast:y:0'Left_eye/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2
Left_eye/reshape/Reshape
%Left_eye/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2'
%Left_eye/conv1d/conv1d/ExpandDims/dimс
!Left_eye/conv1d/conv1d/ExpandDims
ExpandDims!Left_eye/reshape/Reshape:output:0.Left_eye/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd	2#
!Left_eye/conv1d/conv1d/ExpandDimsш
2Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;left_eye_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype024
2Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOpд
(Left_eye/conv1d/conv1d/ExpandDims_1/CastCast:Left_eye/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:	@2*
(Left_eye/conv1d/conv1d/ExpandDims_1/Cast
'Left_eye/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'Left_eye/conv1d/conv1d/ExpandDims_1/dimщ
#Left_eye/conv1d/conv1d/ExpandDims_1
ExpandDims,Left_eye/conv1d/conv1d/ExpandDims_1/Cast:y:00Left_eye/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2%
#Left_eye/conv1d/conv1d/ExpandDims_1ї
Left_eye/conv1d/conv1dConv2D*Left_eye/conv1d/conv1d/ExpandDims:output:0,Left_eye/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@*
paddingVALID*
strides
2
Left_eye/conv1d/conv1dТ
Left_eye/conv1d/conv1d/SqueezeSqueezeLeft_eye/conv1d/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@*
squeeze_dims

§џџџџџџџџ2 
Left_eye/conv1d/conv1d/SqueezeМ
&Left_eye/conv1d/BiasAdd/ReadVariableOpReadVariableOp/left_eye_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&Left_eye/conv1d/BiasAdd/ReadVariableOpЈ
Left_eye/conv1d/BiasAdd/CastCast.Left_eye/conv1d/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
Left_eye/conv1d/BiasAdd/CastО
Left_eye/conv1d/BiasAddBiasAdd'Left_eye/conv1d/conv1d/Squeeze:output:0 Left_eye/conv1d/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
Left_eye/conv1d/BiasAdd
Left_eye/conv1d/ReluRelu Left_eye/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
Left_eye/conv1d/Relu
'Left_eye/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2)
'Left_eye/conv1d_1/conv1d/ExpandDims/dimш
#Left_eye/conv1d_1/conv1d/ExpandDims
ExpandDims"Left_eye/conv1d/Relu:activations:00Left_eye/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@2%
#Left_eye/conv1d_1/conv1d/ExpandDimsю
4Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=left_eye_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype026
4Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpк
*Left_eye/conv1d_1/conv1d/ExpandDims_1/CastCast<Left_eye/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@ 2,
*Left_eye/conv1d_1/conv1d/ExpandDims_1/Cast
)Left_eye/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)Left_eye/conv1d_1/conv1d/ExpandDims_1/dimё
%Left_eye/conv1d_1/conv1d/ExpandDims_1
ExpandDims.Left_eye/conv1d_1/conv1d/ExpandDims_1/Cast:y:02Left_eye/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2'
%Left_eye/conv1d_1/conv1d/ExpandDims_1џ
Left_eye/conv1d_1/conv1dConv2D,Left_eye/conv1d_1/conv1d/ExpandDims:output:0.Left_eye/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
Left_eye/conv1d_1/conv1dШ
 Left_eye/conv1d_1/conv1d/SqueezeSqueeze!Left_eye/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџ2"
 Left_eye/conv1d_1/conv1d/SqueezeТ
(Left_eye/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp1left_eye_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Left_eye/conv1d_1/BiasAdd/ReadVariableOpЎ
Left_eye/conv1d_1/BiasAdd/CastCast0Left_eye/conv1d_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2 
Left_eye/conv1d_1/BiasAdd/CastЦ
Left_eye/conv1d_1/BiasAddBiasAdd)Left_eye/conv1d_1/conv1d/Squeeze:output:0"Left_eye/conv1d_1/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
Left_eye/conv1d_1/BiasAdd
Left_eye/conv1d_1/ReluRelu"Left_eye/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
Left_eye/conv1d_1/Relu
'Left_eye/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2)
'Left_eye/conv1d_2/conv1d/ExpandDims/dimъ
#Left_eye/conv1d_2/conv1d/ExpandDims
ExpandDims$Left_eye/conv1d_1/Relu:activations:00Left_eye/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2%
#Left_eye/conv1d_2/conv1d/ExpandDimsю
4Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=left_eye_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype026
4Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpк
*Left_eye/conv1d_2/conv1d/ExpandDims_1/CastCast<Left_eye/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
: 2,
*Left_eye/conv1d_2/conv1d/ExpandDims_1/Cast
)Left_eye/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2+
)Left_eye/conv1d_2/conv1d/ExpandDims_1/dimё
%Left_eye/conv1d_2/conv1d/ExpandDims_1
ExpandDims.Left_eye/conv1d_2/conv1d/ExpandDims_1/Cast:y:02Left_eye/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2'
%Left_eye/conv1d_2/conv1d/ExpandDims_1џ
Left_eye/conv1d_2/conv1dConv2D,Left_eye/conv1d_2/conv1d/ExpandDims:output:0.Left_eye/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
*
paddingVALID*
strides
2
Left_eye/conv1d_2/conv1dШ
 Left_eye/conv1d_2/conv1d/SqueezeSqueeze!Left_eye/conv1d_2/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
*
squeeze_dims

§џџџџџџџџ2"
 Left_eye/conv1d_2/conv1d/SqueezeТ
(Left_eye/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp1left_eye_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Left_eye/conv1d_2/BiasAdd/ReadVariableOpЎ
Left_eye/conv1d_2/BiasAdd/CastCast0Left_eye/conv1d_2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2 
Left_eye/conv1d_2/BiasAdd/CastЦ
Left_eye/conv1d_2/BiasAddBiasAdd)Left_eye/conv1d_2/conv1d/Squeeze:output:0"Left_eye/conv1d_2/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
Left_eye/conv1d_2/BiasAdd
Left_eye/conv1d_2/ReluRelu"Left_eye/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
Left_eye/conv1d_2/Relu{
Right_eye/CastCastinputs_1*

DstT0*

SrcT0*/
_output_shapes
:џџџџџџџџџd2
Right_eye/Castx
Right_eye/reshape_1/ShapeShapeRight_eye/Cast:y:0*
T0*
_output_shapes
:2
Right_eye/reshape_1/Shape
'Right_eye/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'Right_eye/reshape_1/strided_slice/stack 
)Right_eye/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)Right_eye/reshape_1/strided_slice/stack_1 
)Right_eye/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)Right_eye/reshape_1/strided_slice/stack_2к
!Right_eye/reshape_1/strided_sliceStridedSlice"Right_eye/reshape_1/Shape:output:00Right_eye/reshape_1/strided_slice/stack:output:02Right_eye/reshape_1/strided_slice/stack_1:output:02Right_eye/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!Right_eye/reshape_1/strided_slice
#Right_eye/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d2%
#Right_eye/reshape_1/Reshape/shape/1
#Right_eye/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :	2%
#Right_eye/reshape_1/Reshape/shape/2
!Right_eye/reshape_1/Reshape/shapePack*Right_eye/reshape_1/strided_slice:output:0,Right_eye/reshape_1/Reshape/shape/1:output:0,Right_eye/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!Right_eye/reshape_1/Reshape/shapeЛ
Right_eye/reshape_1/ReshapeReshapeRight_eye/Cast:y:0*Right_eye/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџd	2
Right_eye/reshape_1/Reshape
(Right_eye/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(Right_eye/conv1d_3/conv1d/ExpandDims/dimэ
$Right_eye/conv1d_3/conv1d/ExpandDims
ExpandDims$Right_eye/reshape_1/Reshape:output:01Right_eye/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџd	2&
$Right_eye/conv1d_3/conv1d/ExpandDimsё
5Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>right_eye_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	@*
dtype027
5Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpн
+Right_eye/conv1d_3/conv1d/ExpandDims_1/CastCast=Right_eye/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:	@2-
+Right_eye/conv1d_3/conv1d/ExpandDims_1/Cast
*Right_eye/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*Right_eye/conv1d_3/conv1d/ExpandDims_1/dimѕ
&Right_eye/conv1d_3/conv1d/ExpandDims_1
ExpandDims/Right_eye/conv1d_3/conv1d/ExpandDims_1/Cast:y:03Right_eye/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	@2(
&Right_eye/conv1d_3/conv1d/ExpandDims_1
Right_eye/conv1d_3/conv1dConv2D-Right_eye/conv1d_3/conv1d/ExpandDims:output:0/Right_eye/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@*
paddingVALID*
strides
2
Right_eye/conv1d_3/conv1dЫ
!Right_eye/conv1d_3/conv1d/SqueezeSqueeze"Right_eye/conv1d_3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@*
squeeze_dims

§џџџџџџџџ2#
!Right_eye/conv1d_3/conv1d/SqueezeХ
)Right_eye/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp2right_eye_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)Right_eye/conv1d_3/BiasAdd/ReadVariableOpБ
Right_eye/conv1d_3/BiasAdd/CastCast1Right_eye/conv1d_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2!
Right_eye/conv1d_3/BiasAdd/CastЪ
Right_eye/conv1d_3/BiasAddBiasAdd*Right_eye/conv1d_3/conv1d/Squeeze:output:0#Right_eye/conv1d_3/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
Right_eye/conv1d_3/BiasAdd
Right_eye/conv1d_3/ReluRelu#Right_eye/conv1d_3/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ/@2
Right_eye/conv1d_3/Relu
(Right_eye/conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(Right_eye/conv1d_4/conv1d/ExpandDims/dimю
$Right_eye/conv1d_4/conv1d/ExpandDims
ExpandDims%Right_eye/conv1d_3/Relu:activations:01Right_eye/conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ/@2&
$Right_eye/conv1d_4/conv1d/ExpandDimsё
5Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>right_eye_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype027
5Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpн
+Right_eye/conv1d_4/conv1d/ExpandDims_1/CastCast=Right_eye/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@ 2-
+Right_eye/conv1d_4/conv1d/ExpandDims_1/Cast
*Right_eye/conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*Right_eye/conv1d_4/conv1d/ExpandDims_1/dimѕ
&Right_eye/conv1d_4/conv1d/ExpandDims_1
ExpandDims/Right_eye/conv1d_4/conv1d/ExpandDims_1/Cast:y:03Right_eye/conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2(
&Right_eye/conv1d_4/conv1d/ExpandDims_1
Right_eye/conv1d_4/conv1dConv2D-Right_eye/conv1d_4/conv1d/ExpandDims:output:0/Right_eye/conv1d_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
Right_eye/conv1d_4/conv1dЫ
!Right_eye/conv1d_4/conv1d/SqueezeSqueeze"Right_eye/conv1d_4/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџ2#
!Right_eye/conv1d_4/conv1d/SqueezeХ
)Right_eye/conv1d_4/BiasAdd/ReadVariableOpReadVariableOp2right_eye_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)Right_eye/conv1d_4/BiasAdd/ReadVariableOpБ
Right_eye/conv1d_4/BiasAdd/CastCast1Right_eye/conv1d_4/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
: 2!
Right_eye/conv1d_4/BiasAdd/CastЪ
Right_eye/conv1d_4/BiasAddBiasAdd*Right_eye/conv1d_4/conv1d/Squeeze:output:0#Right_eye/conv1d_4/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
Right_eye/conv1d_4/BiasAdd
Right_eye/conv1d_4/ReluRelu#Right_eye/conv1d_4/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ 2
Right_eye/conv1d_4/Relu
(Right_eye/conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2*
(Right_eye/conv1d_5/conv1d/ExpandDims/dimю
$Right_eye/conv1d_5/conv1d/ExpandDims
ExpandDims%Right_eye/conv1d_4/Relu:activations:01Right_eye/conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2&
$Right_eye/conv1d_5/conv1d/ExpandDimsё
5Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>right_eye_conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype027
5Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpн
+Right_eye/conv1d_5/conv1d/ExpandDims_1/CastCast=Right_eye/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
: 2-
+Right_eye/conv1d_5/conv1d/ExpandDims_1/Cast
*Right_eye/conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*Right_eye/conv1d_5/conv1d/ExpandDims_1/dimѕ
&Right_eye/conv1d_5/conv1d/ExpandDims_1
ExpandDims/Right_eye/conv1d_5/conv1d/ExpandDims_1/Cast:y:03Right_eye/conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2(
&Right_eye/conv1d_5/conv1d/ExpandDims_1
Right_eye/conv1d_5/conv1dConv2D-Right_eye/conv1d_5/conv1d/ExpandDims:output:0/Right_eye/conv1d_5/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
*
paddingVALID*
strides
2
Right_eye/conv1d_5/conv1dЫ
!Right_eye/conv1d_5/conv1d/SqueezeSqueeze"Right_eye/conv1d_5/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
*
squeeze_dims

§џџџџџџџџ2#
!Right_eye/conv1d_5/conv1d/SqueezeХ
)Right_eye/conv1d_5/BiasAdd/ReadVariableOpReadVariableOp2right_eye_conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)Right_eye/conv1d_5/BiasAdd/ReadVariableOpБ
Right_eye/conv1d_5/BiasAdd/CastCast1Right_eye/conv1d_5/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2!
Right_eye/conv1d_5/BiasAdd/CastЪ
Right_eye/conv1d_5/BiasAddBiasAdd*Right_eye/conv1d_5/conv1d/Squeeze:output:0#Right_eye/conv1d_5/BiasAdd/Cast:y:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
Right_eye/conv1d_5/BiasAdd
Right_eye/conv1d_5/ReluRelu#Right_eye/conv1d_5/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
2
Right_eye/conv1d_5/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisт
concatenate/concatConcatV2$Left_eye/conv1d_2/Relu:activations:0%Right_eye/conv1d_5/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ
 2
concatenate/concato
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@  2
flatten/Const
flatten/ReshapeReshapeconcatenate/concat:output:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџР2
flatten/ReshapeЁ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
Р*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMul/CastCast#dense/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:
Р2
dense/MatMul/Cast
dense/MatMulMatMulflatten/Reshape:output:0dense/MatMul/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAdd/CastCast$dense/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:2
dense/BiasAdd/Cast
dense/BiasAddBiasAdddense/MatMul:product:0dense/BiasAdd/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

dense/ReluЇ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMul/CastCast%dense_1/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:
2
dense_1/MatMul/Cast
dense_1/MatMulMatMuldense/Relu:activations:0dense_1/MatMul/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_1/MatMulЅ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAdd/CastCast&dense_1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:2
dense_1/BiasAdd/Cast
dense_1/BiasAddBiasAdddense_1/MatMul:product:0dense_1/BiasAdd/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_1/ReluІ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMul/CastCast%dense_2/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	@2
dense_2/MatMul/Cast
dense_2/MatMulMatMuldense_1/Relu:activations:0dense_2/MatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_2/MatMulЄ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAdd/CastCast&dense_2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@2
dense_2/BiasAdd/Cast
dense_2/BiasAddBiasAdddense_2/MatMul:product:0dense_2/BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@2
dense_2/ReluЅ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMul/CastCast%dense_3/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes

:@2
dense_3/MatMul/Cast
dense_3/MatMulMatMuldense_2/Relu:activations:0dense_3/MatMul/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/MatMulЄ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp
dense_3/BiasAdd/CastCast&dense_3/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:2
dense_3/BiasAdd/Cast
dense_3/BiasAddBiasAdddense_3/MatMul:product:0dense_3/BiasAdd/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/BiasAddo
CastCastdense_3/BiasAdd:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2
Cast\
IdentityIdentityCast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd:џџџџџџџџџd:::::::::::::::::::::Y U
/
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџd
"
_user_specified_name
inputs/1
ж
u
I__inference_concatenate_layer_call_and_return_conditional_losses_33197999
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ
 2
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
 2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:џџџџџџџџџ
:џџџџџџџџџ
:U Q
+
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
inputs/1

Н
+__inference_Left_eye_layer_call_fn_33196511
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_Left_eye_layer_call_and_return_conditional_losses_331964962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:џџџџџџџџџd::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџd
!
_user_specified_name	input_1
Ѓ

&__inference_signature_wrapper_33197310
left	
right
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identityЂStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallleftrightunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *,
f'R%
#__inference__wrapped_model_331962842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*
_input_shapes
:џџџџџџџџџd:џџџџџџџџџd::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
/
_output_shapes
:џџџџџџџџџd

_user_specified_nameLeft:VR
/
_output_shapes
:џџџџџџџџџd

_user_specified_nameRight"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*№
serving_defaultм
=
Left5
serving_default_Left:0џџџџџџџџџd
?
Right6
serving_default_Right:0џџџџџџџџџd>

activation0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ЯБ
ПЂ
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
#_self_saveable_object_factories
	optimizer

signatures
regularization_losses
	variables
trainable_variables
	keras_api
 __call__
+Ё&call_and_return_all_conditional_losses
Ђ_default_save_signature"
_tf_keras_network{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Left"}, "name": "Left", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Right"}, "name": "Right", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "Left_eye", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}, "name": "reshape", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv1d_2", 0, 0]]}, "name": "Left_eye", "inbound_nodes": [[["Left", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "Right_eye", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}, "name": "reshape_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_4", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_4", "inbound_nodes": [[["conv1d_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_5", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_5", "inbound_nodes": [[["conv1d_4", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv1d_5", 0, 0]]}, "name": "Right_eye", "inbound_nodes": [[["Right", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "name": "concatenate", "inbound_nodes": [[["Left_eye", 1, 0, {}], ["Right_eye", 1, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "linear"}, "name": "activation", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}], "input_layers": [["Left", 0, 0], ["Right", 0, 0]], "output_layers": [["activation", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 100, 3, 3]}, {"class_name": "TensorShape", "items": [null, 100, 3, 3]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Left"}, "name": "Left", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Right"}, "name": "Right", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "Left_eye", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}, "name": "reshape", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv1d_2", 0, 0]]}, "name": "Left_eye", "inbound_nodes": [[["Left", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "Right_eye", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}, "name": "reshape_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_4", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_4", "inbound_nodes": [[["conv1d_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_5", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_5", "inbound_nodes": [[["conv1d_4", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv1d_5", 0, 0]]}, "name": "Right_eye", "inbound_nodes": [[["Right", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "name": "concatenate", "inbound_nodes": [[["Left_eye", 1, 0, {}], ["Right_eye", 1, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "linear"}, "name": "activation", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}], "input_layers": [["Left", 0, 0], ["Right", 0, 0]], "output_layers": [["activation", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": ["mean_squared_error"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "LossScaleOptimizer", "config": {"optimizer": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}, "loss_scale": {"class_name": "DynamicLossScale", "config": {"initial_loss_scale": 32768.0, "increment_period": 2000, "multiplier": 2.0}}}}}}

#_self_saveable_object_factories"№
_tf_keras_input_layerа{"class_name": "InputLayer", "name": "Left", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Left"}}

#_self_saveable_object_factories"ђ
_tf_keras_input_layerв{"class_name": "InputLayer", "name": "Right", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Right"}}
У4
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
Ѓ__call__
+Є&call_and_return_all_conditional_losses"ў1
_tf_keras_networkт1{"class_name": "Functional", "name": "Left_eye", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Left_eye", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}, "name": "reshape", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv1d_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 3, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Left_eye", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}, "name": "reshape", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_1", "inbound_nodes": [[["conv1d", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_2", "inbound_nodes": [[["conv1d_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv1d_2", 0, 0]]}}}
о4
layer-0
 layer-1
!layer_with_weights-0
!layer-2
"layer_with_weights-1
"layer-3
#layer_with_weights-2
#layer-4
#$_self_saveable_object_factories
%regularization_losses
&	variables
'trainable_variables
(	keras_api
Ѕ__call__
+І&call_and_return_all_conditional_losses"2
_tf_keras_network§1{"class_name": "Functional", "name": "Right_eye", "trainable": true, "expects_training_arg": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Right_eye", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}, "name": "reshape_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_4", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_4", "inbound_nodes": [[["conv1d_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_5", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_5", "inbound_nodes": [[["conv1d_4", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv1d_5", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 3, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Right_eye", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}, "name": "reshape_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_3", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_3", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_4", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_4", "inbound_nodes": [[["conv1d_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "conv1d_5", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1d_5", "inbound_nodes": [[["conv1d_4", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv1d_5", 0, 0]]}}}
р
#)_self_saveable_object_factories
*regularization_losses
+	variables
,trainable_variables
-	keras_api
Ї__call__
+Ј&call_and_return_all_conditional_losses"Њ
_tf_keras_layer{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 10, 16]}, {"class_name": "TensorShape", "items": [null, 10, 16]}]}
ё
#._self_saveable_object_factories
/regularization_losses
0	variables
1trainable_variables
2	keras_api
Љ__call__
+Њ&call_and_return_all_conditional_losses"Л
_tf_keras_layerЁ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ў

3kernel
4bias
#5_self_saveable_object_factories
6regularization_losses
7	variables
8trainable_variables
9	keras_api
Ћ__call__
+Ќ&call_and_return_all_conditional_losses"В
_tf_keras_layer{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 320}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 320]}}
	

:kernel
;bias
#<_self_saveable_object_factories
=regularization_losses
>	variables
?trainable_variables
@	keras_api
­__call__
+Ў&call_and_return_all_conditional_losses"Ж
_tf_keras_layer{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
	

Akernel
Bbias
#C_self_saveable_object_factories
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
Џ__call__
+А&call_and_return_all_conditional_losses"Е
_tf_keras_layer{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
	

Hkernel
Ibias
#J_self_saveable_object_factories
Kregularization_losses
L	variables
Mtrainable_variables
N	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"Д
_tf_keras_layer{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
њ
#O_self_saveable_object_factories
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"Ф
_tf_keras_layerЊ{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "linear"}}
 "
trackable_dict_wrapper

T
loss_scale
Ubase_optimizer
Viter

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rate3mј4mљ:mњ;mћAmќBm§HmўImџ[m\m]m^m_m`mambmcmdmemfm3v4v:v;vAvBvHvIv[v\v]v^v_v`vavbvcvdvevfv"
	optimizer
-
Еserving_default"
signature_map
 "
trackable_list_wrapper
Ж
[0
\1
]2
^3
_4
`5
a6
b7
c8
d9
e10
f11
312
413
:14
;15
A16
B17
H18
I19"
trackable_list_wrapper
Ж
[0
\1
]2
^3
_4
`5
a6
b7
c8
d9
e10
f11
312
413
:14
;15
A16
B17
H18
I19"
trackable_list_wrapper
Ю
regularization_losses
gmetrics
	variables
hnon_trainable_variables
trainable_variables
ilayer_metrics

jlayers
klayer_regularization_losses
 __call__
Ђ_default_save_signature
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper

#l_self_saveable_object_factories"і
_tf_keras_input_layerж{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}

#m_self_saveable_object_factories
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses"Ы
_tf_keras_layerБ{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}}
№


[kernel
\bias
#r_self_saveable_object_factories
sregularization_losses
t	variables
utrainable_variables
v	keras_api
И__call__
+Й&call_and_return_all_conditional_losses"Є	
_tf_keras_layer	{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 9]}}
ѕ


]kernel
^bias
#w_self_saveable_object_factories
xregularization_losses
y	variables
ztrainable_variables
{	keras_api
К__call__
+Л&call_and_return_all_conditional_losses"Љ	
_tf_keras_layer	{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 47, 64]}}
і


_kernel
`bias
#|_self_saveable_object_factories
}regularization_losses
~	variables
trainable_variables
	keras_api
М__call__
+Н&call_and_return_all_conditional_losses"Љ	
_tf_keras_layer	{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 32]}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
[0
\1
]2
^3
_4
`5"
trackable_list_wrapper
J
[0
\1
]2
^3
_4
`5"
trackable_list_wrapper
Е
regularization_losses
metrics
	variables
non_trainable_variables
trainable_variables
layer_metrics
layers
 layer_regularization_losses
Ѓ__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object

$_self_saveable_object_factories"і
_tf_keras_input_layerж{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}

$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
О__call__
+П&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"class_name": "Reshape", "name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape_1", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "target_shape": {"class_name": "__tuple__", "items": [100, 9]}}}
љ


akernel
bbias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"Ј	
_tf_keras_layer	{"class_name": "Conv1D", "name": "conv1d_3", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_3", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [7]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 9]}}
њ


ckernel
dbias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
Т__call__
+У&call_and_return_all_conditional_losses"Љ	
_tf_keras_layer	{"class_name": "Conv1D", "name": "conv1d_4", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_4", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 47, 64]}}
њ


ekernel
fbias
$_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"Љ	
_tf_keras_layer	{"class_name": "Conv1D", "name": "conv1d_5", "trainable": true, "expects_training_arg": false, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_5", "trainable": true, "dtype": {"class_name": "Policy", "config": {"name": "mixed_float16"}}, "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 32]}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
a0
b1
c2
d3
e4
f5"
trackable_list_wrapper
J
a0
b1
c2
d3
e4
f5"
trackable_list_wrapper
Е
%regularization_losses
metrics
&	variables
non_trainable_variables
'trainable_variables
layer_metrics
layers
 layer_regularization_losses
Ѕ__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
*regularization_losses
 metrics
+	variables
Ёnon_trainable_variables
,trainable_variables
Ђlayer_metrics
Ѓlayers
 Єlayer_regularization_losses
Ї__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
/regularization_losses
Ѕmetrics
0	variables
Іnon_trainable_variables
1trainable_variables
Їlayer_metrics
Јlayers
 Љlayer_regularization_losses
Љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
 :
Р2dense/kernel
:2
dense/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
Е
6regularization_losses
Њmetrics
7	variables
Ћnon_trainable_variables
8trainable_variables
Ќlayer_metrics
­layers
 Ўlayer_regularization_losses
Ћ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_1/kernel
:2dense_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
Е
=regularization_losses
Џmetrics
>	variables
Аnon_trainable_variables
?trainable_variables
Бlayer_metrics
Вlayers
 Гlayer_regularization_losses
­__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
!:	@2dense_2/kernel
:@2dense_2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
Е
Dregularization_losses
Дmetrics
E	variables
Еnon_trainable_variables
Ftrainable_variables
Жlayer_metrics
Зlayers
 Иlayer_regularization_losses
Џ__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_3/kernel
:2dense_3/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
Е
Kregularization_losses
Йmetrics
L	variables
Кnon_trainable_variables
Mtrainable_variables
Лlayer_metrics
Мlayers
 Нlayer_regularization_losses
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Pregularization_losses
Оmetrics
Q	variables
Пnon_trainable_variables
Rtrainable_variables
Рlayer_metrics
Сlayers
 Тlayer_regularization_losses
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
H
Уcurrent_loss_scale
Ф
good_steps"
_generic_user_object
"
_generic_user_object
:	 (2cond_1/Adam/iter
: (2cond_1/Adam/beta_1
: (2cond_1/Adam/beta_2
: (2cond_1/Adam/decay
#:! (2cond_1/Adam/learning_rate
#:!	@2conv1d/kernel
:@2conv1d/bias
%:#@ 2conv1d_1/kernel
: 2conv1d_1/bias
%:# 2conv1d_2/kernel
:2conv1d_2/bias
%:#	@2conv1d_3/kernel
:@2conv1d_3/bias
%:#@ 2conv1d_4/kernel
: 2conv1d_4/bias
%:# 2conv1d_5/kernel
:2conv1d_5/bias
0
Х0
Ц1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
nregularization_losses
Чmetrics
o	variables
Шnon_trainable_variables
ptrainable_variables
Щlayer_metrics
Ъlayers
 Ыlayer_regularization_losses
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
Е
sregularization_losses
Ьmetrics
t	variables
Эnon_trainable_variables
utrainable_variables
Юlayer_metrics
Яlayers
 аlayer_regularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
Е
xregularization_losses
бmetrics
y	variables
вnon_trainable_variables
ztrainable_variables
гlayer_metrics
дlayers
 еlayer_regularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
Е
}regularization_losses
жmetrics
~	variables
зnon_trainable_variables
trainable_variables
иlayer_metrics
йlayers
 кlayer_regularization_losses
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
regularization_losses
лmetrics
	variables
мnon_trainable_variables
trainable_variables
нlayer_metrics
оlayers
 пlayer_regularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
И
regularization_losses
рmetrics
	variables
сnon_trainable_variables
trainable_variables
тlayer_metrics
уlayers
 фlayer_regularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
И
regularization_losses
хmetrics
	variables
цnon_trainable_variables
trainable_variables
чlayer_metrics
шlayers
 щlayer_regularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
И
regularization_losses
ъmetrics
	variables
ыnon_trainable_variables
trainable_variables
ьlayer_metrics
эlayers
 юlayer_regularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
 1
!2
"3
#4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
: 2current_loss_scale
:	 2
good_steps
П

яtotal

№count
ё	variables
ђ	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


ѓtotal

єcount
ѕ
_fn_kwargs
і	variables
ї	keras_api"Ъ
_tf_keras_metricЏ{"class_name": "MeanMetricWrapper", "name": "mean_squared_error", "dtype": "float32", "config": {"name": "mean_squared_error", "dtype": "float32", "fn": "mean_squared_error"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
я0
№1"
trackable_list_wrapper
.
ё	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ѓ0
є1"
trackable_list_wrapper
.
і	variables"
_generic_user_object
,:*
Р2cond_1/Adam/dense/kernel/m
%:#2cond_1/Adam/dense/bias/m
.:,
2cond_1/Adam/dense_1/kernel/m
':%2cond_1/Adam/dense_1/bias/m
-:+	@2cond_1/Adam/dense_2/kernel/m
&:$@2cond_1/Adam/dense_2/bias/m
,:*@2cond_1/Adam/dense_3/kernel/m
&:$2cond_1/Adam/dense_3/bias/m
/:-	@2cond_1/Adam/conv1d/kernel/m
%:#@2cond_1/Adam/conv1d/bias/m
1:/@ 2cond_1/Adam/conv1d_1/kernel/m
':% 2cond_1/Adam/conv1d_1/bias/m
1:/ 2cond_1/Adam/conv1d_2/kernel/m
':%2cond_1/Adam/conv1d_2/bias/m
1:/	@2cond_1/Adam/conv1d_3/kernel/m
':%@2cond_1/Adam/conv1d_3/bias/m
1:/@ 2cond_1/Adam/conv1d_4/kernel/m
':% 2cond_1/Adam/conv1d_4/bias/m
1:/ 2cond_1/Adam/conv1d_5/kernel/m
':%2cond_1/Adam/conv1d_5/bias/m
,:*
Р2cond_1/Adam/dense/kernel/v
%:#2cond_1/Adam/dense/bias/v
.:,
2cond_1/Adam/dense_1/kernel/v
':%2cond_1/Adam/dense_1/bias/v
-:+	@2cond_1/Adam/dense_2/kernel/v
&:$@2cond_1/Adam/dense_2/bias/v
,:*@2cond_1/Adam/dense_3/kernel/v
&:$2cond_1/Adam/dense_3/bias/v
/:-	@2cond_1/Adam/conv1d/kernel/v
%:#@2cond_1/Adam/conv1d/bias/v
1:/@ 2cond_1/Adam/conv1d_1/kernel/v
':% 2cond_1/Adam/conv1d_1/bias/v
1:/ 2cond_1/Adam/conv1d_2/kernel/v
':%2cond_1/Adam/conv1d_2/bias/v
1:/	@2cond_1/Adam/conv1d_3/kernel/v
':%@2cond_1/Adam/conv1d_3/bias/v
1:/@ 2cond_1/Adam/conv1d_4/kernel/v
':% 2cond_1/Adam/conv1d_4/bias/v
1:/ 2cond_1/Adam/conv1d_5/kernel/v
':%2cond_1/Adam/conv1d_5/bias/v
2
/__inference_functional_1_layer_call_fn_33197153
/__inference_functional_1_layer_call_fn_33197654
/__inference_functional_1_layer_call_fn_33197700
/__inference_functional_1_layer_call_fn_33197254Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
і2ѓ
J__inference_functional_1_layer_call_and_return_conditional_losses_33197459
J__inference_functional_1_layer_call_and_return_conditional_losses_33197608
J__inference_functional_1_layer_call_and_return_conditional_losses_33197051
J__inference_functional_1_layer_call_and_return_conditional_losses_33196996Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
#__inference__wrapped_model_33196284щ
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *YЂV
TQ
&#
Leftџџџџџџџџџd
'$
Rightџџџџџџџџџd
њ2ї
+__inference_Left_eye_layer_call_fn_33197829
+__inference_Left_eye_layer_call_fn_33197846
+__inference_Left_eye_layer_call_fn_33196473
+__inference_Left_eye_layer_call_fn_33196511Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
F__inference_Left_eye_layer_call_and_return_conditional_losses_33197756
F__inference_Left_eye_layer_call_and_return_conditional_losses_33197812
F__inference_Left_eye_layer_call_and_return_conditional_losses_33196434
F__inference_Left_eye_layer_call_and_return_conditional_losses_33196413Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ў2ћ
,__inference_Right_eye_layer_call_fn_33197992
,__inference_Right_eye_layer_call_fn_33196700
,__inference_Right_eye_layer_call_fn_33197975
,__inference_Right_eye_layer_call_fn_33196738Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ъ2ч
G__inference_Right_eye_layer_call_and_return_conditional_losses_33197902
G__inference_Right_eye_layer_call_and_return_conditional_losses_33197958
G__inference_Right_eye_layer_call_and_return_conditional_losses_33196640
G__inference_Right_eye_layer_call_and_return_conditional_losses_33196661Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
и2е
.__inference_concatenate_layer_call_fn_33198005Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓ2№
I__inference_concatenate_layer_call_and_return_conditional_losses_33197999Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_flatten_layer_call_fn_33198016Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_flatten_layer_call_and_return_conditional_losses_33198011Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_dense_layer_call_fn_33198038Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_dense_layer_call_and_return_conditional_losses_33198029Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_1_layer_call_fn_33198060Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_1_layer_call_and_return_conditional_losses_33198051Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_2_layer_call_fn_33198082Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_2_layer_call_and_return_conditional_losses_33198073Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_3_layer_call_fn_33198103Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_3_layer_call_and_return_conditional_losses_33198094Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_activation_layer_call_fn_33198112Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_activation_layer_call_and_return_conditional_losses_33198107Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
7B5
&__inference_signature_wrapper_33197310LeftRight
д2б
*__inference_reshape_layer_call_fn_33198130Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_reshape_layer_call_and_return_conditional_losses_33198125Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv1d_layer_call_fn_33198157Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv1d_layer_call_and_return_conditional_losses_33198148Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_conv1d_1_layer_call_fn_33198184Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_conv1d_1_layer_call_and_return_conditional_losses_33198175Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_conv1d_2_layer_call_fn_33198211Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_conv1d_2_layer_call_and_return_conditional_losses_33198202Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_reshape_1_layer_call_fn_33198229Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_reshape_1_layer_call_and_return_conditional_losses_33198224Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_conv1d_3_layer_call_fn_33198256Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_conv1d_3_layer_call_and_return_conditional_losses_33198247Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_conv1d_4_layer_call_fn_33198283Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_conv1d_4_layer_call_and_return_conditional_losses_33198274Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_conv1d_5_layer_call_fn_33198310Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_conv1d_5_layer_call_and_return_conditional_losses_33198301Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 П
F__inference_Left_eye_layer_call_and_return_conditional_losses_33196413u[\]^_`@Ђ=
6Ђ3
)&
input_1џџџџџџџџџd
p

 
Њ ")Ђ&

0џџџџџџџџџ

 П
F__inference_Left_eye_layer_call_and_return_conditional_losses_33196434u[\]^_`@Ђ=
6Ђ3
)&
input_1џџџџџџџџџd
p 

 
Њ ")Ђ&

0џџџџџџџџџ

 О
F__inference_Left_eye_layer_call_and_return_conditional_losses_33197756t[\]^_`?Ђ<
5Ђ2
(%
inputsџџџџџџџџџd
p

 
Њ ")Ђ&

0џџџџџџџџџ

 О
F__inference_Left_eye_layer_call_and_return_conditional_losses_33197812t[\]^_`?Ђ<
5Ђ2
(%
inputsџџџџџџџџџd
p 

 
Њ ")Ђ&

0џџџџџџџџџ

 
+__inference_Left_eye_layer_call_fn_33196473h[\]^_`@Ђ=
6Ђ3
)&
input_1џџџџџџџџџd
p

 
Њ "џџџџџџџџџ

+__inference_Left_eye_layer_call_fn_33196511h[\]^_`@Ђ=
6Ђ3
)&
input_1џџџџџџџџџd
p 

 
Њ "џџџџџџџџџ

+__inference_Left_eye_layer_call_fn_33197829g[\]^_`?Ђ<
5Ђ2
(%
inputsџџџџџџџџџd
p

 
Њ "џџџџџџџџџ

+__inference_Left_eye_layer_call_fn_33197846g[\]^_`?Ђ<
5Ђ2
(%
inputsџџџџџџџџџd
p 

 
Њ "џџџџџџџџџ
Р
G__inference_Right_eye_layer_call_and_return_conditional_losses_33196640uabcdef@Ђ=
6Ђ3
)&
input_2џџџџџџџџџd
p

 
Њ ")Ђ&

0џџџџџџџџџ

 Р
G__inference_Right_eye_layer_call_and_return_conditional_losses_33196661uabcdef@Ђ=
6Ђ3
)&
input_2џџџџџџџџџd
p 

 
Њ ")Ђ&

0џџџџџџџџџ

 П
G__inference_Right_eye_layer_call_and_return_conditional_losses_33197902tabcdef?Ђ<
5Ђ2
(%
inputsџџџџџџџџџd
p

 
Њ ")Ђ&

0џџџџџџџџџ

 П
G__inference_Right_eye_layer_call_and_return_conditional_losses_33197958tabcdef?Ђ<
5Ђ2
(%
inputsџџџџџџџџџd
p 

 
Њ ")Ђ&

0џџџџџџџџџ

 
,__inference_Right_eye_layer_call_fn_33196700habcdef@Ђ=
6Ђ3
)&
input_2џџџџџџџџџd
p

 
Њ "џџџџџџџџџ

,__inference_Right_eye_layer_call_fn_33196738habcdef@Ђ=
6Ђ3
)&
input_2џџџџџџџџџd
p 

 
Њ "џџџџџџџџџ

,__inference_Right_eye_layer_call_fn_33197975gabcdef?Ђ<
5Ђ2
(%
inputsџџџџџџџџџd
p

 
Њ "џџџџџџџџџ

,__inference_Right_eye_layer_call_fn_33197992gabcdef?Ђ<
5Ђ2
(%
inputsџџџџџџџџџd
p 

 
Њ "џџџџџџџџџ
м
#__inference__wrapped_model_33196284Д[\]^_`abcdef34:;ABHIcЂ`
YЂV
TQ
&#
Leftџџџџџџџџџd
'$
Rightџџџџџџџџџd
Њ "7Њ4
2

activation$!

activationџџџџџџџџџЄ
H__inference_activation_layer_call_and_return_conditional_losses_33198107X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 |
-__inference_activation_layer_call_fn_33198112K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџн
I__inference_concatenate_layer_call_and_return_conditional_losses_33197999bЂ_
XЂU
SP
&#
inputs/0џџџџџџџџџ

&#
inputs/1џџџџџџџџџ

Њ ")Ђ&

0џџџџџџџџџ
 
 Е
.__inference_concatenate_layer_call_fn_33198005bЂ_
XЂU
SP
&#
inputs/0џџџџџџџџџ

&#
inputs/1џџџџџџџџџ

Њ "џџџџџџџџџ
 Ў
F__inference_conv1d_1_layer_call_and_return_conditional_losses_33198175d]^3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ/@
Њ ")Ђ&

0џџџџџџџџџ 
 
+__inference_conv1d_1_layer_call_fn_33198184W]^3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ/@
Њ "џџџџџџџџџ Ў
F__inference_conv1d_2_layer_call_and_return_conditional_losses_33198202d_`3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ 
Њ ")Ђ&

0џџџџџџџџџ

 
+__inference_conv1d_2_layer_call_fn_33198211W_`3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ
Ў
F__inference_conv1d_3_layer_call_and_return_conditional_losses_33198247dab3Ђ0
)Ђ&
$!
inputsџџџџџџџџџd	
Њ ")Ђ&

0џџџџџџџџџ/@
 
+__inference_conv1d_3_layer_call_fn_33198256Wab3Ђ0
)Ђ&
$!
inputsџџџџџџџџџd	
Њ "џџџџџџџџџ/@Ў
F__inference_conv1d_4_layer_call_and_return_conditional_losses_33198274dcd3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ/@
Њ ")Ђ&

0џџџџџџџџџ 
 
+__inference_conv1d_4_layer_call_fn_33198283Wcd3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ/@
Њ "џџџџџџџџџ Ў
F__inference_conv1d_5_layer_call_and_return_conditional_losses_33198301def3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ 
Њ ")Ђ&

0џџџџџџџџџ

 
+__inference_conv1d_5_layer_call_fn_33198310Wef3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ
Ќ
D__inference_conv1d_layer_call_and_return_conditional_losses_33198148d[\3Ђ0
)Ђ&
$!
inputsџџџџџџџџџd	
Њ ")Ђ&

0џџџџџџџџџ/@
 
)__inference_conv1d_layer_call_fn_33198157W[\3Ђ0
)Ђ&
$!
inputsџџџџџџџџџd	
Њ "џџџџџџџџџ/@Ї
E__inference_dense_1_layer_call_and_return_conditional_losses_33198051^:;0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_dense_1_layer_call_fn_33198060Q:;0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
E__inference_dense_2_layer_call_and_return_conditional_losses_33198073]AB0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ@
 ~
*__inference_dense_2_layer_call_fn_33198082PAB0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ@Ѕ
E__inference_dense_3_layer_call_and_return_conditional_losses_33198094\HI/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_dense_3_layer_call_fn_33198103OHI/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџЅ
C__inference_dense_layer_call_and_return_conditional_losses_33198029^340Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "&Ђ#

0џџџџџџџџџ
 }
(__inference_dense_layer_call_fn_33198038Q340Ђ-
&Ђ#
!
inputsџџџџџџџџџР
Њ "џџџџџџџџџІ
E__inference_flatten_layer_call_and_return_conditional_losses_33198011]3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
 
Њ "&Ђ#

0џџџџџџџџџР
 ~
*__inference_flatten_layer_call_fn_33198016P3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
 
Њ "џџџџџџџџџРљ
J__inference_functional_1_layer_call_and_return_conditional_losses_33196996Њ[\]^_`abcdef34:;ABHIkЂh
aЂ^
TQ
&#
Leftџџџџџџџџџd
'$
Rightџџџџџџџџџd
p

 
Њ "%Ђ"

0џџџџџџџџџ
 љ
J__inference_functional_1_layer_call_and_return_conditional_losses_33197051Њ[\]^_`abcdef34:;ABHIkЂh
aЂ^
TQ
&#
Leftџџџџџџџџџd
'$
Rightџџџџџџџџџd
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 
J__inference_functional_1_layer_call_and_return_conditional_losses_33197459Б[\]^_`abcdef34:;ABHIrЂo
hЂe
[X
*'
inputs/0џџџџџџџџџd
*'
inputs/1џџџџџџџџџd
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
J__inference_functional_1_layer_call_and_return_conditional_losses_33197608Б[\]^_`abcdef34:;ABHIrЂo
hЂe
[X
*'
inputs/0џџџџџџџџџd
*'
inputs/1џџџџџџџџџd
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 б
/__inference_functional_1_layer_call_fn_33197153[\]^_`abcdef34:;ABHIkЂh
aЂ^
TQ
&#
Leftџџџџџџџџџd
'$
Rightџџџџџџџџџd
p

 
Њ "џџџџџџџџџб
/__inference_functional_1_layer_call_fn_33197254[\]^_`abcdef34:;ABHIkЂh
aЂ^
TQ
&#
Leftџџџџџџџџџd
'$
Rightџџџџџџџџџd
p 

 
Њ "џџџџџџџџџи
/__inference_functional_1_layer_call_fn_33197654Є[\]^_`abcdef34:;ABHIrЂo
hЂe
[X
*'
inputs/0џџџџџџџџџd
*'
inputs/1џџџџџџџџџd
p

 
Њ "џџџџџџџџџи
/__inference_functional_1_layer_call_fn_33197700Є[\]^_`abcdef34:;ABHIrЂo
hЂe
[X
*'
inputs/0џџџџџџџџџd
*'
inputs/1џџџџџџџџџd
p 

 
Њ "џџџџџџџџџЏ
G__inference_reshape_1_layer_call_and_return_conditional_losses_33198224d7Ђ4
-Ђ*
(%
inputsџџџџџџџџџd
Њ ")Ђ&

0џџџџџџџџџd	
 
,__inference_reshape_1_layer_call_fn_33198229W7Ђ4
-Ђ*
(%
inputsџџџџџџџџџd
Њ "џџџџџџџџџd	­
E__inference_reshape_layer_call_and_return_conditional_losses_33198125d7Ђ4
-Ђ*
(%
inputsџџџџџџџџџd
Њ ")Ђ&

0џџџџџџџџџd	
 
*__inference_reshape_layer_call_fn_33198130W7Ђ4
-Ђ*
(%
inputsџџџџџџџџџd
Њ "џџџџџџџџџd	ы
&__inference_signature_wrapper_33197310Р[\]^_`abcdef34:;ABHIoЂl
Ђ 
eЊb
.
Left&#
Leftџџџџџџџџџd
0
Right'$
Rightџџџџџџџџџd"7Њ4
2

activation$!

activationџџџџџџџџџ
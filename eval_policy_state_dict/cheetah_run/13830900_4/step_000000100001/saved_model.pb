В╣
┐Б
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
dtypetypeѕ
Й
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
executor_typestring ѕ
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.2.02unknown8вд
Ћ
feedforward_mlp_torso/linear/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*/
shared_name feedforward_mlp_torso/linear/b
ј
2feedforward_mlp_torso/linear/b/Read/ReadVariableOpReadVariableOpfeedforward_mlp_torso/linear/b*
_output_shapes	
:ђ*
dtype0
Ў
feedforward_mlp_torso/linear/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*/
shared_name feedforward_mlp_torso/linear/w
њ
2feedforward_mlp_torso/linear/w/Read/ReadVariableOpReadVariableOpfeedforward_mlp_torso/linear/w*
_output_shapes
:	ђ*
dtype0
Д
'feedforward_mlp_torso/layer_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*8
shared_name)'feedforward_mlp_torso/layer_norm/offset
а
;feedforward_mlp_torso/layer_norm/offset/Read/ReadVariableOpReadVariableOp'feedforward_mlp_torso/layer_norm/offset*
_output_shapes	
:ђ*
dtype0
Ц
&feedforward_mlp_torso/layer_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*7
shared_name(&feedforward_mlp_torso/layer_norm/scale
ъ
:feedforward_mlp_torso/layer_norm/scale/Read/ReadVariableOpReadVariableOp&feedforward_mlp_torso/layer_norm/scale*
_output_shapes	
:ђ*
dtype0
А
$feedforward_mlp_torso/mlp/linear_0/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*5
shared_name&$feedforward_mlp_torso/mlp/linear_0/b
џ
8feedforward_mlp_torso/mlp/linear_0/b/Read/ReadVariableOpReadVariableOp$feedforward_mlp_torso/mlp/linear_0/b*
_output_shapes	
:ђ*
dtype0
д
$feedforward_mlp_torso/mlp/linear_0/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*5
shared_name&$feedforward_mlp_torso/mlp/linear_0/w
Ъ
8feedforward_mlp_torso/mlp/linear_0/w/Read/ReadVariableOpReadVariableOp$feedforward_mlp_torso/mlp/linear_0/w* 
_output_shapes
:
ђђ*
dtype0
А
$feedforward_mlp_torso/mlp/linear_1/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*5
shared_name&$feedforward_mlp_torso/mlp/linear_1/b
џ
8feedforward_mlp_torso/mlp/linear_1/b/Read/ReadVariableOpReadVariableOp$feedforward_mlp_torso/mlp/linear_1/b*
_output_shapes	
:ђ*
dtype0
д
$feedforward_mlp_torso/mlp/linear_1/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ђђ*5
shared_name&$feedforward_mlp_torso/mlp/linear_1/w
Ъ
8feedforward_mlp_torso/mlp/linear_1/w/Read/ReadVariableOpReadVariableOp$feedforward_mlp_torso/mlp/linear_1/w* 
_output_shapes
:
ђђ*
dtype0
ћ
near_zero_initialized_linear/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name near_zero_initialized_linear/b
Ї
2near_zero_initialized_linear/b/Read/ReadVariableOpReadVariableOpnear_zero_initialized_linear/b*
_output_shapes
:*
dtype0
Ў
near_zero_initialized_linear/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*/
shared_name near_zero_initialized_linear/w
њ
2near_zero_initialized_linear/w/Read/ReadVariableOpReadVariableOpnear_zero_initialized_linear/w*
_output_shapes
:	ђ*
dtype0

NoOpNoOp
Ы	
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Г	
valueБ	Bа	 BЎ	
:

_variables
_trainable_variables

signatures
F
0
1
2
3
4
	5

6
7
8
9
F
0
1
2
3
4
	5

6
7
8
9
 
[Y
VARIABLE_VALUEfeedforward_mlp_torso/linear/b'_variables/0/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEfeedforward_mlp_torso/linear/w'_variables/1/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'feedforward_mlp_torso/layer_norm/offset'_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&feedforward_mlp_torso/layer_norm/scale'_variables/3/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$feedforward_mlp_torso/mlp/linear_0/b'_variables/4/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$feedforward_mlp_torso/mlp/linear_0/w'_variables/5/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$feedforward_mlp_torso/mlp/linear_1/b'_variables/6/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$feedforward_mlp_torso/mlp/linear_1/w'_variables/7/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEnear_zero_initialized_linear/b'_variables/8/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEnear_zero_initialized_linear/w'_variables/9/.ATTRIBUTES/VARIABLE_VALUE
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
н
StatefulPartitionedCallStatefulPartitionedCallsaver_filename2feedforward_mlp_torso/linear/b/Read/ReadVariableOp2feedforward_mlp_torso/linear/w/Read/ReadVariableOp;feedforward_mlp_torso/layer_norm/offset/Read/ReadVariableOp:feedforward_mlp_torso/layer_norm/scale/Read/ReadVariableOp8feedforward_mlp_torso/mlp/linear_0/b/Read/ReadVariableOp8feedforward_mlp_torso/mlp/linear_0/w/Read/ReadVariableOp8feedforward_mlp_torso/mlp/linear_1/b/Read/ReadVariableOp8feedforward_mlp_torso/mlp/linear_1/w/Read/ReadVariableOp2near_zero_initialized_linear/b/Read/ReadVariableOp2near_zero_initialized_linear/w/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__traced_save_285482
Ѕ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamefeedforward_mlp_torso/linear/bfeedforward_mlp_torso/linear/w'feedforward_mlp_torso/layer_norm/offset&feedforward_mlp_torso/layer_norm/scale$feedforward_mlp_torso/mlp/linear_0/b$feedforward_mlp_torso/mlp/linear_0/w$feedforward_mlp_torso/mlp/linear_1/b$feedforward_mlp_torso/mlp/linear_1/wnear_zero_initialized_linear/bnear_zero_initialized_linear/w*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__traced_restore_285524╔э
ѓ
ё
__inference___call___5976
args_0_position
args_0_velocity
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityѕбStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallargs_0_positionargs_0_velocityunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference_wrapped_module_9662
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:         :         	::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:         
)
_user_specified_nameargs_0/position:XT
'
_output_shapes
:         	
)
_user_specified_nameargs_0/velocity:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
Бє
ё
__inference_wrapped_module_6066
args_0_position
args_0_velocity?
;feedforward_mlp_torso_linear_matmul_readvariableop_resource<
8feedforward_mlp_torso_linear_add_readvariableop_resourceJ
Ffeedforward_mlp_torso_layer_norm_batchnorm_mul_readvariableop_resourceF
Bfeedforward_mlp_torso_layer_norm_batchnorm_readvariableop_resourceE
Afeedforward_mlp_torso_mlp_linear_0_matmul_readvariableop_resourceB
>feedforward_mlp_torso_mlp_linear_0_add_readvariableop_resourceE
Afeedforward_mlp_torso_mlp_linear_1_matmul_readvariableop_resourceB
>feedforward_mlp_torso_mlp_linear_1_add_readvariableop_resource?
;near_zero_initialized_linear_matmul_readvariableop_resource<
8near_zero_initialized_linear_add_readvariableop_resource
identityѕs
sequential/flatten/ShapeShapeargs_0_position*
T0*
_output_shapes
:2
sequential/flatten/Shapeџ
&sequential/flatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/flatten/strided_slice/stackъ
(sequential/flatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/flatten/strided_slice/stack_1ъ
(sequential/flatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/flatten/strided_slice/stack_2м
 sequential/flatten/strided_sliceStridedSlice!sequential/flatten/Shape:output:0/sequential/flatten/strided_slice/stack:output:01sequential/flatten/strided_slice/stack_1:output:01sequential/flatten/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2"
 sequential/flatten/strided_sliceњ
"sequential/flatten/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"sequential/flatten/concat/values_1ѓ
sequential/flatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
sequential/flatten/concat/axisы
sequential/flatten/concatConcatV2)sequential/flatten/strided_slice:output:0+sequential/flatten/concat/values_1:output:0'sequential/flatten/concat/axis:output:0*
N*
T0*
_output_shapes
:2
sequential/flatten/concatф
sequential/flatten/ReshapeReshapeargs_0_position"sequential/flatten/concat:output:0*
T0*'
_output_shapes
:         2
sequential/flatten/Reshapew
sequential/flatten/Shape_1Shapeargs_0_velocity*
T0*
_output_shapes
:2
sequential/flatten/Shape_1ъ
(sequential/flatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential/flatten/strided_slice_1/stackб
*sequential/flatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential/flatten/strided_slice_1/stack_1б
*sequential/flatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential/flatten/strided_slice_1/stack_2я
"sequential/flatten/strided_slice_1StridedSlice#sequential/flatten/Shape_1:output:01sequential/flatten/strided_slice_1/stack:output:03sequential/flatten/strided_slice_1/stack_1:output:03sequential/flatten/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2$
"sequential/flatten/strided_slice_1ќ
$sequential/flatten/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:	2&
$sequential/flatten/concat_1/values_1є
 sequential/flatten/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential/flatten/concat_1/axisч
sequential/flatten/concat_1ConcatV2+sequential/flatten/strided_slice_1:output:0-sequential/flatten/concat_1/values_1:output:0)sequential/flatten/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
sequential/flatten/concat_1░
sequential/flatten/Reshape_1Reshapeargs_0_velocity$sequential/flatten/concat_1:output:0*
T0*'
_output_shapes
:         	2
sequential/flatten/Reshape_1{
sequential/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         2
sequential/concat/axis┌
sequential/concatConcatV2#sequential/flatten/Reshape:output:0%sequential/flatten/Reshape_1:output:0sequential/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
sequential/concatћ
#feedforward_mlp_torso/flatten/ShapeShapesequential/concat:output:0*
T0*
_output_shapes
:2%
#feedforward_mlp_torso/flatten/Shape░
1feedforward_mlp_torso/flatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1feedforward_mlp_torso/flatten/strided_slice/stack┤
3feedforward_mlp_torso/flatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3feedforward_mlp_torso/flatten/strided_slice/stack_1┤
3feedforward_mlp_torso/flatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3feedforward_mlp_torso/flatten/strided_slice/stack_2ћ
+feedforward_mlp_torso/flatten/strided_sliceStridedSlice,feedforward_mlp_torso/flatten/Shape:output:0:feedforward_mlp_torso/flatten/strided_slice/stack:output:0<feedforward_mlp_torso/flatten/strided_slice/stack_1:output:0<feedforward_mlp_torso/flatten/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+feedforward_mlp_torso/flatten/strided_sliceе
-feedforward_mlp_torso/flatten/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-feedforward_mlp_torso/flatten/concat/values_1ў
)feedforward_mlp_torso/flatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)feedforward_mlp_torso/flatten/concat/axisе
$feedforward_mlp_torso/flatten/concatConcatV24feedforward_mlp_torso/flatten/strided_slice:output:06feedforward_mlp_torso/flatten/concat/values_1:output:02feedforward_mlp_torso/flatten/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$feedforward_mlp_torso/flatten/concatо
%feedforward_mlp_torso/flatten/ReshapeReshapesequential/concat:output:0-feedforward_mlp_torso/flatten/concat:output:0*
T0*'
_output_shapes
:         2'
%feedforward_mlp_torso/flatten/ReshapeЮ
'feedforward_mlp_torso/concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
         2)
'feedforward_mlp_torso/concat/concat_dimИ
#feedforward_mlp_torso/concat/concatIdentity.feedforward_mlp_torso/flatten/Reshape:output:0*
T0*'
_output_shapes
:         2%
#feedforward_mlp_torso/concat/concatт
2feedforward_mlp_torso/linear/MatMul/ReadVariableOpReadVariableOp;feedforward_mlp_torso_linear_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype024
2feedforward_mlp_torso/linear/MatMul/ReadVariableOpы
#feedforward_mlp_torso/linear/MatMulMatMul,feedforward_mlp_torso/concat/concat:output:0:feedforward_mlp_torso/linear/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2%
#feedforward_mlp_torso/linear/MatMulп
/feedforward_mlp_torso/linear/Add/ReadVariableOpReadVariableOp8feedforward_mlp_torso_linear_add_readvariableop_resource*
_output_shapes	
:ђ*
dtype021
/feedforward_mlp_torso/linear/Add/ReadVariableOpТ
 feedforward_mlp_torso/linear/AddAdd-feedforward_mlp_torso/linear/MatMul:product:07feedforward_mlp_torso/linear/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2"
 feedforward_mlp_torso/linear/Add╠
?feedforward_mlp_torso/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2A
?feedforward_mlp_torso/layer_norm/moments/mean/reduction_indicesЎ
-feedforward_mlp_torso/layer_norm/moments/meanMean$feedforward_mlp_torso/linear/Add:z:0Hfeedforward_mlp_torso/layer_norm/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2/
-feedforward_mlp_torso/layer_norm/moments/meanУ
5feedforward_mlp_torso/layer_norm/moments/StopGradientStopGradient6feedforward_mlp_torso/layer_norm/moments/mean:output:0*
T0*'
_output_shapes
:         27
5feedforward_mlp_torso/layer_norm/moments/StopGradientд
:feedforward_mlp_torso/layer_norm/moments/SquaredDifferenceSquaredDifference$feedforward_mlp_torso/linear/Add:z:0>feedforward_mlp_torso/layer_norm/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ2<
:feedforward_mlp_torso/layer_norm/moments/SquaredDifferenceн
Cfeedforward_mlp_torso/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2E
Cfeedforward_mlp_torso/layer_norm/moments/variance/reduction_indices┐
1feedforward_mlp_torso/layer_norm/moments/varianceMean>feedforward_mlp_torso/layer_norm/moments/SquaredDifference:z:0Lfeedforward_mlp_torso/layer_norm/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(23
1feedforward_mlp_torso/layer_norm/moments/varianceЕ
0feedforward_mlp_torso/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *г┼'722
0feedforward_mlp_torso/layer_norm/batchnorm/add/yњ
.feedforward_mlp_torso/layer_norm/batchnorm/addAddV2:feedforward_mlp_torso/layer_norm/moments/variance:output:09feedforward_mlp_torso/layer_norm/batchnorm/add/y:output:0*
T0*'
_output_shapes
:         20
.feedforward_mlp_torso/layer_norm/batchnorm/addМ
0feedforward_mlp_torso/layer_norm/batchnorm/RsqrtRsqrt2feedforward_mlp_torso/layer_norm/batchnorm/add:z:0*
T0*'
_output_shapes
:         22
0feedforward_mlp_torso/layer_norm/batchnorm/Rsqrtѓ
=feedforward_mlp_torso/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpFfeedforward_mlp_torso_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02?
=feedforward_mlp_torso/layer_norm/batchnorm/mul/ReadVariableOpЌ
.feedforward_mlp_torso/layer_norm/batchnorm/mulMul4feedforward_mlp_torso/layer_norm/batchnorm/Rsqrt:y:0Efeedforward_mlp_torso/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ20
.feedforward_mlp_torso/layer_norm/batchnorm/mulЭ
0feedforward_mlp_torso/layer_norm/batchnorm/mul_1Mul$feedforward_mlp_torso/linear/Add:z:02feedforward_mlp_torso/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ22
0feedforward_mlp_torso/layer_norm/batchnorm/mul_1і
0feedforward_mlp_torso/layer_norm/batchnorm/mul_2Mul6feedforward_mlp_torso/layer_norm/moments/mean:output:02feedforward_mlp_torso/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ22
0feedforward_mlp_torso/layer_norm/batchnorm/mul_2Ш
9feedforward_mlp_torso/layer_norm/batchnorm/ReadVariableOpReadVariableOpBfeedforward_mlp_torso_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02;
9feedforward_mlp_torso/layer_norm/batchnorm/ReadVariableOpЊ
.feedforward_mlp_torso/layer_norm/batchnorm/subSubAfeedforward_mlp_torso/layer_norm/batchnorm/ReadVariableOp:value:04feedforward_mlp_torso/layer_norm/batchnorm/mul_2:z:0*
T0*(
_output_shapes
:         ђ20
.feedforward_mlp_torso/layer_norm/batchnorm/subі
0feedforward_mlp_torso/layer_norm/batchnorm/add_1AddV24feedforward_mlp_torso/layer_norm/batchnorm/mul_1:z:02feedforward_mlp_torso/layer_norm/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ22
0feedforward_mlp_torso/layer_norm/batchnorm/add_1┐
%feedforward_mlp_torso/sequential/TanhTanh4feedforward_mlp_torso/layer_norm/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         ђ2'
%feedforward_mlp_torso/sequential/TanhЭ
8feedforward_mlp_torso/mlp/linear_0/MatMul/ReadVariableOpReadVariableOpAfeedforward_mlp_torso_mlp_linear_0_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02:
8feedforward_mlp_torso/mlp/linear_0/MatMul/ReadVariableOpђ
)feedforward_mlp_torso/mlp/linear_0/MatMulMatMul)feedforward_mlp_torso/sequential/Tanh:y:0@feedforward_mlp_torso/mlp/linear_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2+
)feedforward_mlp_torso/mlp/linear_0/MatMulЖ
5feedforward_mlp_torso/mlp/linear_0/Add/ReadVariableOpReadVariableOp>feedforward_mlp_torso_mlp_linear_0_add_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5feedforward_mlp_torso/mlp/linear_0/Add/ReadVariableOp■
&feedforward_mlp_torso/mlp/linear_0/AddAdd3feedforward_mlp_torso/mlp/linear_0/MatMul:product:0=feedforward_mlp_torso/mlp/linear_0/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2(
&feedforward_mlp_torso/mlp/linear_0/Addц
feedforward_mlp_torso/mlp/EluElu*feedforward_mlp_torso/mlp/linear_0/Add:z:0*
T0*(
_output_shapes
:         ђ2
feedforward_mlp_torso/mlp/EluЭ
8feedforward_mlp_torso/mlp/linear_1/MatMul/ReadVariableOpReadVariableOpAfeedforward_mlp_torso_mlp_linear_1_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02:
8feedforward_mlp_torso/mlp/linear_1/MatMul/ReadVariableOpѓ
)feedforward_mlp_torso/mlp/linear_1/MatMulMatMul+feedforward_mlp_torso/mlp/Elu:activations:0@feedforward_mlp_torso/mlp/linear_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2+
)feedforward_mlp_torso/mlp/linear_1/MatMulЖ
5feedforward_mlp_torso/mlp/linear_1/Add/ReadVariableOpReadVariableOp>feedforward_mlp_torso_mlp_linear_1_add_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5feedforward_mlp_torso/mlp/linear_1/Add/ReadVariableOp■
&feedforward_mlp_torso/mlp/linear_1/AddAdd3feedforward_mlp_torso/mlp/linear_1/MatMul:product:0=feedforward_mlp_torso/mlp/linear_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2(
&feedforward_mlp_torso/mlp/linear_1/Addе
feedforward_mlp_torso/mlp/Elu_1Elu*feedforward_mlp_torso/mlp/linear_1/Add:z:0*
T0*(
_output_shapes
:         ђ2!
feedforward_mlp_torso/mlp/Elu_1т
2near_zero_initialized_linear/MatMul/ReadVariableOpReadVariableOp;near_zero_initialized_linear_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype024
2near_zero_initialized_linear/MatMul/ReadVariableOpы
#near_zero_initialized_linear/MatMulMatMul-feedforward_mlp_torso/mlp/Elu_1:activations:0:near_zero_initialized_linear/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2%
#near_zero_initialized_linear/MatMulО
/near_zero_initialized_linear/Add/ReadVariableOpReadVariableOp8near_zero_initialized_linear_add_readvariableop_resource*
_output_shapes
:*
dtype021
/near_zero_initialized_linear/Add/ReadVariableOpт
 near_zero_initialized_linear/AddAdd-near_zero_initialized_linear/MatMul:product:07near_zero_initialized_linear/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2"
 near_zero_initialized_linear/Addє
tanh_to_spec/TanhTanh$near_zero_initialized_linear/Add:z:0*
T0*'
_output_shapes
:         2
tanh_to_spec/Tanhm
tanh_to_spec/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
tanh_to_spec/add/yЊ
tanh_to_spec/addAddV2tanh_to_spec/Tanh:y:0tanh_to_spec/add/y:output:0*
T0*'
_output_shapes
:         2
tanh_to_spec/addm
tanh_to_spec/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
tanh_to_spec/mul/xљ
tanh_to_spec/mulMultanh_to_spec/mul/x:output:0tanh_to_spec/add:z:0*
T0*'
_output_shapes
:         2
tanh_to_spec/mulЇ
tanh_to_spec/mul_1/yConst*
_output_shapes
:*
dtype0*-
value$B""   @   @   @   @   @   @2
tanh_to_spec/mul_1/yќ
tanh_to_spec/mul_1Multanh_to_spec/mul:z:0tanh_to_spec/mul_1/y:output:0*
T0*'
_output_shapes
:         2
tanh_to_spec/mul_1Ї
tanh_to_spec/add_1/yConst*
_output_shapes
:*
dtype0*-
value$B""  ђ┐  ђ┐  ђ┐  ђ┐  ђ┐  ђ┐2
tanh_to_spec/add_1/yџ
tanh_to_spec/add_1AddV2tanh_to_spec/mul_1:z:0tanh_to_spec/add_1/y:output:0*
T0*'
_output_shapes
:         2
tanh_to_spec/add_1j
IdentityIdentitytanh_to_spec/add_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:         :         	:::::::::::X T
'
_output_shapes
:         
)
_user_specified_nameargs_0/position:XT
'
_output_shapes
:         	
)
_user_specified_nameargs_0/velocity:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
ЯЁ
з
__inference_wrapped_module_966

args_0
args_0_1?
;feedforward_mlp_torso_linear_matmul_readvariableop_resource<
8feedforward_mlp_torso_linear_add_readvariableop_resourceJ
Ffeedforward_mlp_torso_layer_norm_batchnorm_mul_readvariableop_resourceF
Bfeedforward_mlp_torso_layer_norm_batchnorm_readvariableop_resourceE
Afeedforward_mlp_torso_mlp_linear_0_matmul_readvariableop_resourceB
>feedforward_mlp_torso_mlp_linear_0_add_readvariableop_resourceE
Afeedforward_mlp_torso_mlp_linear_1_matmul_readvariableop_resourceB
>feedforward_mlp_torso_mlp_linear_1_add_readvariableop_resource?
;near_zero_initialized_linear_matmul_readvariableop_resource<
8near_zero_initialized_linear_add_readvariableop_resource
identityѕj
sequential/flatten/ShapeShapeargs_0*
T0*
_output_shapes
:2
sequential/flatten/Shapeџ
&sequential/flatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/flatten/strided_slice/stackъ
(sequential/flatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/flatten/strided_slice/stack_1ъ
(sequential/flatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential/flatten/strided_slice/stack_2м
 sequential/flatten/strided_sliceStridedSlice!sequential/flatten/Shape:output:0/sequential/flatten/strided_slice/stack:output:01sequential/flatten/strided_slice/stack_1:output:01sequential/flatten/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2"
 sequential/flatten/strided_sliceњ
"sequential/flatten/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"sequential/flatten/concat/values_1ѓ
sequential/flatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
sequential/flatten/concat/axisы
sequential/flatten/concatConcatV2)sequential/flatten/strided_slice:output:0+sequential/flatten/concat/values_1:output:0'sequential/flatten/concat/axis:output:0*
N*
T0*
_output_shapes
:2
sequential/flatten/concatА
sequential/flatten/ReshapeReshapeargs_0"sequential/flatten/concat:output:0*
T0*'
_output_shapes
:         2
sequential/flatten/Reshapep
sequential/flatten/Shape_1Shapeargs_0_1*
T0*
_output_shapes
:2
sequential/flatten/Shape_1ъ
(sequential/flatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential/flatten/strided_slice_1/stackб
*sequential/flatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential/flatten/strided_slice_1/stack_1б
*sequential/flatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential/flatten/strided_slice_1/stack_2я
"sequential/flatten/strided_slice_1StridedSlice#sequential/flatten/Shape_1:output:01sequential/flatten/strided_slice_1/stack:output:03sequential/flatten/strided_slice_1/stack_1:output:03sequential/flatten/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2$
"sequential/flatten/strided_slice_1ќ
$sequential/flatten/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:	2&
$sequential/flatten/concat_1/values_1є
 sequential/flatten/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential/flatten/concat_1/axisч
sequential/flatten/concat_1ConcatV2+sequential/flatten/strided_slice_1:output:0-sequential/flatten/concat_1/values_1:output:0)sequential/flatten/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
sequential/flatten/concat_1Е
sequential/flatten/Reshape_1Reshapeargs_0_1$sequential/flatten/concat_1:output:0*
T0*'
_output_shapes
:         	2
sequential/flatten/Reshape_1{
sequential/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         2
sequential/concat/axis┌
sequential/concatConcatV2#sequential/flatten/Reshape:output:0%sequential/flatten/Reshape_1:output:0sequential/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
sequential/concatћ
#feedforward_mlp_torso/flatten/ShapeShapesequential/concat:output:0*
T0*
_output_shapes
:2%
#feedforward_mlp_torso/flatten/Shape░
1feedforward_mlp_torso/flatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1feedforward_mlp_torso/flatten/strided_slice/stack┤
3feedforward_mlp_torso/flatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3feedforward_mlp_torso/flatten/strided_slice/stack_1┤
3feedforward_mlp_torso/flatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3feedforward_mlp_torso/flatten/strided_slice/stack_2ћ
+feedforward_mlp_torso/flatten/strided_sliceStridedSlice,feedforward_mlp_torso/flatten/Shape:output:0:feedforward_mlp_torso/flatten/strided_slice/stack:output:0<feedforward_mlp_torso/flatten/strided_slice/stack_1:output:0<feedforward_mlp_torso/flatten/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2-
+feedforward_mlp_torso/flatten/strided_sliceе
-feedforward_mlp_torso/flatten/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-feedforward_mlp_torso/flatten/concat/values_1ў
)feedforward_mlp_torso/flatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)feedforward_mlp_torso/flatten/concat/axisе
$feedforward_mlp_torso/flatten/concatConcatV24feedforward_mlp_torso/flatten/strided_slice:output:06feedforward_mlp_torso/flatten/concat/values_1:output:02feedforward_mlp_torso/flatten/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$feedforward_mlp_torso/flatten/concatо
%feedforward_mlp_torso/flatten/ReshapeReshapesequential/concat:output:0-feedforward_mlp_torso/flatten/concat:output:0*
T0*'
_output_shapes
:         2'
%feedforward_mlp_torso/flatten/ReshapeЮ
'feedforward_mlp_torso/concat/concat_dimConst*
_output_shapes
: *
dtype0*
valueB :
         2)
'feedforward_mlp_torso/concat/concat_dimИ
#feedforward_mlp_torso/concat/concatIdentity.feedforward_mlp_torso/flatten/Reshape:output:0*
T0*'
_output_shapes
:         2%
#feedforward_mlp_torso/concat/concatт
2feedforward_mlp_torso/linear/MatMul/ReadVariableOpReadVariableOp;feedforward_mlp_torso_linear_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype024
2feedforward_mlp_torso/linear/MatMul/ReadVariableOpы
#feedforward_mlp_torso/linear/MatMulMatMul,feedforward_mlp_torso/concat/concat:output:0:feedforward_mlp_torso/linear/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2%
#feedforward_mlp_torso/linear/MatMulп
/feedforward_mlp_torso/linear/Add/ReadVariableOpReadVariableOp8feedforward_mlp_torso_linear_add_readvariableop_resource*
_output_shapes	
:ђ*
dtype021
/feedforward_mlp_torso/linear/Add/ReadVariableOpТ
 feedforward_mlp_torso/linear/AddAdd-feedforward_mlp_torso/linear/MatMul:product:07feedforward_mlp_torso/linear/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2"
 feedforward_mlp_torso/linear/Add╠
?feedforward_mlp_torso/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2A
?feedforward_mlp_torso/layer_norm/moments/mean/reduction_indicesЎ
-feedforward_mlp_torso/layer_norm/moments/meanMean$feedforward_mlp_torso/linear/Add:z:0Hfeedforward_mlp_torso/layer_norm/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2/
-feedforward_mlp_torso/layer_norm/moments/meanУ
5feedforward_mlp_torso/layer_norm/moments/StopGradientStopGradient6feedforward_mlp_torso/layer_norm/moments/mean:output:0*
T0*'
_output_shapes
:         27
5feedforward_mlp_torso/layer_norm/moments/StopGradientд
:feedforward_mlp_torso/layer_norm/moments/SquaredDifferenceSquaredDifference$feedforward_mlp_torso/linear/Add:z:0>feedforward_mlp_torso/layer_norm/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ђ2<
:feedforward_mlp_torso/layer_norm/moments/SquaredDifferenceн
Cfeedforward_mlp_torso/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2E
Cfeedforward_mlp_torso/layer_norm/moments/variance/reduction_indices┐
1feedforward_mlp_torso/layer_norm/moments/varianceMean>feedforward_mlp_torso/layer_norm/moments/SquaredDifference:z:0Lfeedforward_mlp_torso/layer_norm/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(23
1feedforward_mlp_torso/layer_norm/moments/varianceЕ
0feedforward_mlp_torso/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *г┼'722
0feedforward_mlp_torso/layer_norm/batchnorm/add/yњ
.feedforward_mlp_torso/layer_norm/batchnorm/addAddV2:feedforward_mlp_torso/layer_norm/moments/variance:output:09feedforward_mlp_torso/layer_norm/batchnorm/add/y:output:0*
T0*'
_output_shapes
:         20
.feedforward_mlp_torso/layer_norm/batchnorm/addМ
0feedforward_mlp_torso/layer_norm/batchnorm/RsqrtRsqrt2feedforward_mlp_torso/layer_norm/batchnorm/add:z:0*
T0*'
_output_shapes
:         22
0feedforward_mlp_torso/layer_norm/batchnorm/Rsqrtѓ
=feedforward_mlp_torso/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpFfeedforward_mlp_torso_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:ђ*
dtype02?
=feedforward_mlp_torso/layer_norm/batchnorm/mul/ReadVariableOpЌ
.feedforward_mlp_torso/layer_norm/batchnorm/mulMul4feedforward_mlp_torso/layer_norm/batchnorm/Rsqrt:y:0Efeedforward_mlp_torso/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ20
.feedforward_mlp_torso/layer_norm/batchnorm/mulЭ
0feedforward_mlp_torso/layer_norm/batchnorm/mul_1Mul$feedforward_mlp_torso/linear/Add:z:02feedforward_mlp_torso/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ22
0feedforward_mlp_torso/layer_norm/batchnorm/mul_1і
0feedforward_mlp_torso/layer_norm/batchnorm/mul_2Mul6feedforward_mlp_torso/layer_norm/moments/mean:output:02feedforward_mlp_torso/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ђ22
0feedforward_mlp_torso/layer_norm/batchnorm/mul_2Ш
9feedforward_mlp_torso/layer_norm/batchnorm/ReadVariableOpReadVariableOpBfeedforward_mlp_torso_layer_norm_batchnorm_readvariableop_resource*
_output_shapes	
:ђ*
dtype02;
9feedforward_mlp_torso/layer_norm/batchnorm/ReadVariableOpЊ
.feedforward_mlp_torso/layer_norm/batchnorm/subSubAfeedforward_mlp_torso/layer_norm/batchnorm/ReadVariableOp:value:04feedforward_mlp_torso/layer_norm/batchnorm/mul_2:z:0*
T0*(
_output_shapes
:         ђ20
.feedforward_mlp_torso/layer_norm/batchnorm/subі
0feedforward_mlp_torso/layer_norm/batchnorm/add_1AddV24feedforward_mlp_torso/layer_norm/batchnorm/mul_1:z:02feedforward_mlp_torso/layer_norm/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ђ22
0feedforward_mlp_torso/layer_norm/batchnorm/add_1┐
%feedforward_mlp_torso/sequential/TanhTanh4feedforward_mlp_torso/layer_norm/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         ђ2'
%feedforward_mlp_torso/sequential/TanhЭ
8feedforward_mlp_torso/mlp/linear_0/MatMul/ReadVariableOpReadVariableOpAfeedforward_mlp_torso_mlp_linear_0_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02:
8feedforward_mlp_torso/mlp/linear_0/MatMul/ReadVariableOpђ
)feedforward_mlp_torso/mlp/linear_0/MatMulMatMul)feedforward_mlp_torso/sequential/Tanh:y:0@feedforward_mlp_torso/mlp/linear_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2+
)feedforward_mlp_torso/mlp/linear_0/MatMulЖ
5feedforward_mlp_torso/mlp/linear_0/Add/ReadVariableOpReadVariableOp>feedforward_mlp_torso_mlp_linear_0_add_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5feedforward_mlp_torso/mlp/linear_0/Add/ReadVariableOp■
&feedforward_mlp_torso/mlp/linear_0/AddAdd3feedforward_mlp_torso/mlp/linear_0/MatMul:product:0=feedforward_mlp_torso/mlp/linear_0/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2(
&feedforward_mlp_torso/mlp/linear_0/Addц
feedforward_mlp_torso/mlp/EluElu*feedforward_mlp_torso/mlp/linear_0/Add:z:0*
T0*(
_output_shapes
:         ђ2
feedforward_mlp_torso/mlp/EluЭ
8feedforward_mlp_torso/mlp/linear_1/MatMul/ReadVariableOpReadVariableOpAfeedforward_mlp_torso_mlp_linear_1_matmul_readvariableop_resource* 
_output_shapes
:
ђђ*
dtype02:
8feedforward_mlp_torso/mlp/linear_1/MatMul/ReadVariableOpѓ
)feedforward_mlp_torso/mlp/linear_1/MatMulMatMul+feedforward_mlp_torso/mlp/Elu:activations:0@feedforward_mlp_torso/mlp/linear_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2+
)feedforward_mlp_torso/mlp/linear_1/MatMulЖ
5feedforward_mlp_torso/mlp/linear_1/Add/ReadVariableOpReadVariableOp>feedforward_mlp_torso_mlp_linear_1_add_readvariableop_resource*
_output_shapes	
:ђ*
dtype027
5feedforward_mlp_torso/mlp/linear_1/Add/ReadVariableOp■
&feedforward_mlp_torso/mlp/linear_1/AddAdd3feedforward_mlp_torso/mlp/linear_1/MatMul:product:0=feedforward_mlp_torso/mlp/linear_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2(
&feedforward_mlp_torso/mlp/linear_1/Addе
feedforward_mlp_torso/mlp/Elu_1Elu*feedforward_mlp_torso/mlp/linear_1/Add:z:0*
T0*(
_output_shapes
:         ђ2!
feedforward_mlp_torso/mlp/Elu_1т
2near_zero_initialized_linear/MatMul/ReadVariableOpReadVariableOp;near_zero_initialized_linear_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype024
2near_zero_initialized_linear/MatMul/ReadVariableOpы
#near_zero_initialized_linear/MatMulMatMul-feedforward_mlp_torso/mlp/Elu_1:activations:0:near_zero_initialized_linear/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2%
#near_zero_initialized_linear/MatMulО
/near_zero_initialized_linear/Add/ReadVariableOpReadVariableOp8near_zero_initialized_linear_add_readvariableop_resource*
_output_shapes
:*
dtype021
/near_zero_initialized_linear/Add/ReadVariableOpт
 near_zero_initialized_linear/AddAdd-near_zero_initialized_linear/MatMul:product:07near_zero_initialized_linear/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2"
 near_zero_initialized_linear/Addє
tanh_to_spec/TanhTanh$near_zero_initialized_linear/Add:z:0*
T0*'
_output_shapes
:         2
tanh_to_spec/Tanhm
tanh_to_spec/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
tanh_to_spec/add/yЊ
tanh_to_spec/addAddV2tanh_to_spec/Tanh:y:0tanh_to_spec/add/y:output:0*
T0*'
_output_shapes
:         2
tanh_to_spec/addm
tanh_to_spec/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
tanh_to_spec/mul/xљ
tanh_to_spec/mulMultanh_to_spec/mul/x:output:0tanh_to_spec/add:z:0*
T0*'
_output_shapes
:         2
tanh_to_spec/mulЇ
tanh_to_spec/mul_1/yConst*
_output_shapes
:*
dtype0*-
value$B""   @   @   @   @   @   @2
tanh_to_spec/mul_1/yќ
tanh_to_spec/mul_1Multanh_to_spec/mul:z:0tanh_to_spec/mul_1/y:output:0*
T0*'
_output_shapes
:         2
tanh_to_spec/mul_1Ї
tanh_to_spec/add_1/yConst*
_output_shapes
:*
dtype0*-
value$B""  ђ┐  ђ┐  ђ┐  ђ┐  ђ┐  ђ┐2
tanh_to_spec/add_1/yџ
tanh_to_spec/add_1AddV2tanh_to_spec/mul_1:z:0tanh_to_spec/add_1/y:output:0*
T0*'
_output_shapes
:         2
tanh_to_spec/add_1j
IdentityIdentitytanh_to_spec/add_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:         :         	:::::::::::O K
'
_output_shapes
:         
 
_user_specified_nameargs_0:OK
'
_output_shapes
:         	
 
_user_specified_nameargs_0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
ц+
А
__inference__traced_save_285482
file_prefix=
9savev2_feedforward_mlp_torso_linear_b_read_readvariableop=
9savev2_feedforward_mlp_torso_linear_w_read_readvariableopF
Bsavev2_feedforward_mlp_torso_layer_norm_offset_read_readvariableopE
Asavev2_feedforward_mlp_torso_layer_norm_scale_read_readvariableopC
?savev2_feedforward_mlp_torso_mlp_linear_0_b_read_readvariableopC
?savev2_feedforward_mlp_torso_mlp_linear_0_w_read_readvariableopC
?savev2_feedforward_mlp_torso_mlp_linear_1_b_read_readvariableopC
?savev2_feedforward_mlp_torso_mlp_linear_1_w_read_readvariableop=
9savev2_near_zero_initialized_linear_b_read_readvariableop=
9savev2_near_zero_initialized_linear_w_read_readvariableop
savev2_1_const

identity_1ѕбMergeV2CheckpointsбSaveV2бSaveV2_1Ј
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
ConstЇ
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_c7e94bd7543749868cd25148088f16c5/part2	
Const_1І
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЮ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*»
valueЦBб
B'_variables/0/.ATTRIBUTES/VARIABLE_VALUEB'_variables/1/.ATTRIBUTES/VARIABLE_VALUEB'_variables/2/.ATTRIBUTES/VARIABLE_VALUEB'_variables/3/.ATTRIBUTES/VARIABLE_VALUEB'_variables/4/.ATTRIBUTES/VARIABLE_VALUEB'_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'_variables/6/.ATTRIBUTES/VARIABLE_VALUEB'_variables/7/.ATTRIBUTES/VARIABLE_VALUEB'_variables/8/.ATTRIBUTES/VARIABLE_VALUEB'_variables/9/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesю
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
SaveV2/shape_and_slicesг
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:09savev2_feedforward_mlp_torso_linear_b_read_readvariableop9savev2_feedforward_mlp_torso_linear_w_read_readvariableopBsavev2_feedforward_mlp_torso_layer_norm_offset_read_readvariableopAsavev2_feedforward_mlp_torso_layer_norm_scale_read_readvariableop?savev2_feedforward_mlp_torso_mlp_linear_0_b_read_readvariableop?savev2_feedforward_mlp_torso_mlp_linear_0_w_read_readvariableop?savev2_feedforward_mlp_torso_mlp_linear_1_b_read_readvariableop?savev2_feedforward_mlp_torso_mlp_linear_1_w_read_readvariableop9savev2_near_zero_initialized_linear_b_read_readvariableop9savev2_near_zero_initialized_linear_w_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2
2
SaveV2Ѓ
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardг
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1б
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesј
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices¤
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1с
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesг
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityЂ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*n
_input_shapes]
[: :ђ:	ђ:ђ:ђ:ђ:
ђђ:ђ:
ђђ::	ђ: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:!

_output_shapes	
:ђ:&"
 
_output_shapes
:
ђђ:!

_output_shapes	
:ђ:&"
 
_output_shapes
:
ђђ: 	

_output_shapes
::%
!

_output_shapes
:	ђ:

_output_shapes
: 
Ј4
ђ
"__inference__traced_restore_285524
file_prefix3
/assignvariableop_feedforward_mlp_torso_linear_b5
1assignvariableop_1_feedforward_mlp_torso_linear_w>
:assignvariableop_2_feedforward_mlp_torso_layer_norm_offset=
9assignvariableop_3_feedforward_mlp_torso_layer_norm_scale;
7assignvariableop_4_feedforward_mlp_torso_mlp_linear_0_b;
7assignvariableop_5_feedforward_mlp_torso_mlp_linear_0_w;
7assignvariableop_6_feedforward_mlp_torso_mlp_linear_1_b;
7assignvariableop_7_feedforward_mlp_torso_mlp_linear_1_w5
1assignvariableop_8_near_zero_initialized_linear_b5
1assignvariableop_9_near_zero_initialized_linear_w
identity_11ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9б	RestoreV2бRestoreV2_1Б
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*»
valueЦBб
B'_variables/0/.ATTRIBUTES/VARIABLE_VALUEB'_variables/1/.ATTRIBUTES/VARIABLE_VALUEB'_variables/2/.ATTRIBUTES/VARIABLE_VALUEB'_variables/3/.ATTRIBUTES/VARIABLE_VALUEB'_variables/4/.ATTRIBUTES/VARIABLE_VALUEB'_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'_variables/6/.ATTRIBUTES/VARIABLE_VALUEB'_variables/7/.ATTRIBUTES/VARIABLE_VALUEB'_variables/8/.ATTRIBUTES/VARIABLE_VALUEB'_variables/9/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesб
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
RestoreV2/shape_and_slicesП
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*<
_output_shapes*
(::::::::::*
dtypes
2
2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityЪ
AssignVariableOpAssignVariableOp/assignvariableop_feedforward_mlp_torso_linear_bIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Д
AssignVariableOp_1AssignVariableOp1assignvariableop_1_feedforward_mlp_torso_linear_wIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2░
AssignVariableOp_2AssignVariableOp:assignvariableop_2_feedforward_mlp_torso_layer_norm_offsetIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3»
AssignVariableOp_3AssignVariableOp9assignvariableop_3_feedforward_mlp_torso_layer_norm_scaleIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Г
AssignVariableOp_4AssignVariableOp7assignvariableop_4_feedforward_mlp_torso_mlp_linear_0_bIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Г
AssignVariableOp_5AssignVariableOp7assignvariableop_5_feedforward_mlp_torso_mlp_linear_0_wIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Г
AssignVariableOp_6AssignVariableOp7assignvariableop_6_feedforward_mlp_torso_mlp_linear_1_bIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Г
AssignVariableOp_7AssignVariableOp7assignvariableop_7_feedforward_mlp_torso_mlp_linear_1_wIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8Д
AssignVariableOp_8AssignVariableOp1assignvariableop_8_near_zero_initialized_linear_bIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Д
AssignVariableOp_9AssignVariableOp1assignvariableop_9_near_zero_initialized_linear_wIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9е
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesћ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp║
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10К
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: "ЮJ
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:І
l

_variables
_trainable_variables

signatures
__call__
_module"
acme_snapshot
g
0
1
2
3
4
	5

6
7
8
9"
trackable_tuple_wrapper
g
0
1
2
3
4
	5

6
7
8
9"
trackable_tuple_wrapper
"
signature_map
-:+ђ2feedforward_mlp_torso/linear/b
1:/	ђ2feedforward_mlp_torso/linear/w
6:4ђ2'feedforward_mlp_torso/layer_norm/offset
5:3ђ2&feedforward_mlp_torso/layer_norm/scale
3:1ђ2$feedforward_mlp_torso/mlp/linear_0/b
8:6
ђђ2$feedforward_mlp_torso/mlp/linear_0/w
3:1ђ2$feedforward_mlp_torso/mlp/linear_1/b
8:6
ђђ2$feedforward_mlp_torso/mlp/linear_1/w
,:*2near_zero_initialized_linear/b
1:/	ђ2near_zero_initialized_linear/w
├2└
__inference___call___5976б
Ў▓Ћ
FullArgSpec
argsџ
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┐2╝
__inference_wrapped_module_6066ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 К
__inference___call___5976Е
	
ђб}
vбs
qфn
5
position)і&
args_0/position         
5
velocity)і&
args_0/velocity         	
ф "і         ═
__inference_wrapped_module_6066Е
	
ђб}
vбs
qфn
5
position)і&
args_0/position         
5
velocity)і&
args_0/velocity         	
ф "і         
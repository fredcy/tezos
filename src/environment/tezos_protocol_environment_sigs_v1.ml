module type T = sig
module Pervasives : sig
# 1 "environment/v1/pervasives.mli"
(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*             Xavier Leroy, projet Cristal, INRIA Rocquencourt           *)
(*                                                                        *)
(*   Copyright 1996 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* TEZOS CHANGES

   * Import version 4.04.0
   * Remove [channel], [exit], ...
   * Remove polymorphic comparisons
   * Remove non IEEE754-standardized functions on floats
   * Remove deprecated functions

*)


(** The initially opened module.

   This module provides the basic operations over the built-in types
   (numbers, booleans, byte sequences, strings, exceptions, references,
   lists, arrays, input-output channels, ...).

   This module is automatically opened at the beginning of each compilation.
   All components of this module can therefore be referred by their short
   name, without prefixing them by [Pervasives].
*)


(** {6 Exceptions} *)

external raise : exn -> 'a = "%raise"
(** Raise the given exception value *)

external raise_notrace : exn -> 'a = "%raise_notrace"
(** A faster version [raise] which does not record the backtrace.
    @since 4.02.0
*)

val invalid_arg : string -> 'a
(** Raise exception [Invalid_argument] with the given string. *)

val failwith : string -> 'a
(** Raise exception [Failure] with the given string. *)

exception Exit
(** The [Exit] exception is not raised by any library function.  It is
    provided for use in your programs. *)


(** {6 Boolean operations} *)

external not : bool -> bool = "%boolnot"
(** The boolean negation. *)

external ( && ) : bool -> bool -> bool = "%sequand"
(** The boolean 'and'. Evaluation is sequential, left-to-right:
   in [e1 && e2], [e1] is evaluated first, and if it returns [false],
   [e2] is not evaluated at all. *)


external ( || ) : bool -> bool -> bool = "%sequor"
(** The boolean 'or'. Evaluation is sequential, left-to-right:
   in [e1 || e2], [e1] is evaluated first, and if it returns [true],
   [e2] is not evaluated at all. *)

(** {6 Debugging} *)

external __LOC__ : string = "%loc_LOC"
(** [__LOC__] returns the location at which this expression appears in
    the file currently being parsed by the compiler, with the standard
    error format of OCaml: "File %S, line %d, characters %d-%d".
    @since 4.02.0
*)

external __FILE__ : string = "%loc_FILE"
(** [__FILE__] returns the name of the file currently being
    parsed by the compiler.
    @since 4.02.0
*)

external __LINE__ : int = "%loc_LINE"
(** [__LINE__] returns the line number at which this expression
    appears in the file currently being parsed by the compiler.
    @since 4.02.0
*)

external __MODULE__ : string = "%loc_MODULE"
(** [__MODULE__] returns the module name of the file being
    parsed by the compiler.
    @since 4.02.0
*)

external __POS__ : string * int * int * int = "%loc_POS"
(** [__POS__] returns a tuple [(file,lnum,cnum,enum)], corresponding
    to the location at which this expression appears in the file
    currently being parsed by the compiler. [file] is the current
    filename, [lnum] the line number, [cnum] the character position in
    the line and [enum] the last character position in the line.
    @since 4.02.0
 *)

external __LOC_OF__ : 'a -> string * 'a = "%loc_LOC"
(** [__LOC_OF__ expr] returns a pair [(loc, expr)] where [loc] is the
    location of [expr] in the file currently being parsed by the
    compiler, with the standard error format of OCaml: "File %S, line
    %d, characters %d-%d".
    @since 4.02.0
*)

external __LINE_OF__ : 'a -> int * 'a = "%loc_LINE"
(** [__LINE__ expr] returns a pair [(line, expr)], where [line] is the
    line number at which the expression [expr] appears in the file
    currently being parsed by the compiler.
    @since 4.02.0
 *)

external __POS_OF__ : 'a -> (string * int * int * int) * 'a = "%loc_POS"
(** [__POS_OF__ expr] returns a pair [(loc,expr)], where [loc] is a
    tuple [(file,lnum,cnum,enum)] corresponding to the location at
    which the expression [expr] appears in the file currently being
    parsed by the compiler. [file] is the current filename, [lnum] the
    line number, [cnum] the character position in the line and [enum]
    the last character position in the line.
    @since 4.02.0
 *)

(** {6 Composition operators} *)

external ( |> ) : 'a -> ('a -> 'b) -> 'b = "%revapply"
(** Reverse-application operator: [x |> f |> g] is exactly equivalent
 to [g (f (x))].
   @since 4.01
*)

external ( @@ ) : ('a -> 'b) -> 'a -> 'b = "%apply"
(** Application operator: [g @@ f @@ x] is exactly equivalent to
 [g (f (x))].
   @since 4.01
*)

(** {6 Integer arithmetic} *)

(** Integers are 31 bits wide (or 63 bits on 64-bit processors).
   All operations are taken modulo 2{^31} (or 2{^63}).
   They do not fail on overflow. *)

external ( ~- ) : int -> int = "%negint"
(** Unary negation. You can also write [- e] instead of [~- e]. *)

external ( ~+ ) : int -> int = "%identity"
(** Unary addition. You can also write [+ e] instead of [~+ e].
    @since 3.12.0
*)

external succ : int -> int = "%succint"
(** [succ x] is [x + 1]. *)

external pred : int -> int = "%predint"
(** [pred x] is [x - 1]. *)

external ( + ) : int -> int -> int = "%addint"
(** Integer addition. *)

external ( - ) : int -> int -> int = "%subint"
(** Integer subtraction. *)

external ( * ) : int -> int -> int = "%mulint"
(** Integer multiplication. *)

external ( / ) : int -> int -> int = "%divint"
(** Integer division.
   Raise [Division_by_zero] if the second argument is 0.
   Integer division rounds the real quotient of its arguments towards zero.
   More precisely, if [x >= 0] and [y > 0], [x / y] is the greatest integer
   less than or equal to the real quotient of [x] by [y].  Moreover,
   [(- x) / y = x / (- y) = - (x / y)].  *)

external ( mod ) : int -> int -> int = "%modint"
(** Integer remainder.  If [y] is not zero, the result
   of [x mod y] satisfies the following properties:
   [x = (x / y) * y + x mod y] and
   [abs(x mod y) <= abs(y) - 1].
   If [y = 0], [x mod y] raises [Division_by_zero].
   Note that [x mod y] is negative only if [x < 0].
   Raise [Division_by_zero] if [y] is zero. *)

val abs : int -> int
(** Return the absolute value of the argument.  Note that this may be
  negative if the argument is [min_int]. *)

val max_int : int
(** The greatest representable integer. *)

val min_int : int
(** The smallest representable integer. *)


(** {7 Bitwise operations} *)

external ( land ) : int -> int -> int = "%andint"
(** Bitwise logical and. *)

external ( lor ) : int -> int -> int = "%orint"
(** Bitwise logical or. *)

external ( lxor ) : int -> int -> int = "%xorint"
(** Bitwise logical exclusive or. *)

val lnot : int -> int
(** Bitwise logical negation. *)

external ( lsl ) : int -> int -> int = "%lslint"
(** [n lsl m] shifts [n] to the left by [m] bits.
   The result is unspecified if [m < 0] or [m >= bitsize],
   where [bitsize] is [32] on a 32-bit platform and
   [64] on a 64-bit platform. *)

external ( lsr ) : int -> int -> int = "%lsrint"
(** [n lsr m] shifts [n] to the right by [m] bits.
   This is a logical shift: zeroes are inserted regardless of
   the sign of [n].
   The result is unspecified if [m < 0] or [m >= bitsize]. *)

external ( asr ) : int -> int -> int = "%asrint"
(** [n asr m] shifts [n] to the right by [m] bits.
   This is an arithmetic shift: the sign bit of [n] is replicated.
   The result is unspecified if [m < 0] or [m >= bitsize]. *)


(** {6 Floating-point arithmetic}

   OCaml's floating-point numbers follow the
   IEEE 754 standard, using double precision (64 bits) numbers.
   Floating-point operations never raise an exception on overflow,
   underflow, division by zero, etc.  Instead, special IEEE numbers
   are returned as appropriate, such as [infinity] for [1.0 /. 0.0],
   [neg_infinity] for [-1.0 /. 0.0], and [nan] ('not a number')
   for [0.0 /. 0.0].  These special numbers then propagate through
   floating-point computations as expected: for instance,
   [1.0 /. infinity] is [0.0], and any arithmetic operation with [nan]
   as argument returns [nan] as result.
*)

external ( ~-. ) : float -> float = "%negfloat"
(** Unary negation. You can also write [-. e] instead of [~-. e]. *)

external ( ~+. ) : float -> float = "%identity"
(** Unary addition. You can also write [+. e] instead of [~+. e].
    @since 3.12.0
*)

external ( +. ) : float -> float -> float = "%addfloat"
(** Floating-point addition *)

external ( -. ) : float -> float -> float = "%subfloat"
(** Floating-point subtraction *)

external ( *. ) : float -> float -> float = "%mulfloat"
(** Floating-point multiplication *)

external ( /. ) : float -> float -> float = "%divfloat"
(** Floating-point division. *)

external ceil : float -> float = "caml_ceil_float" "ceil"
  [@@unboxed] [@@noalloc]
(** Round above to an integer value.
    [ceil f] returns the least integer value greater than or equal to [f].
    The result is returned as a float. *)

external floor : float -> float = "caml_floor_float" "floor"
  [@@unboxed] [@@noalloc]
(** Round below to an integer value.
    [floor f] returns the greatest integer value less than or
    equal to [f].
    The result is returned as a float. *)

external abs_float : float -> float = "%absfloat"
(** [abs_float f] returns the absolute value of [f]. *)

external copysign : float -> float -> float
                  = "caml_copysign_float" "caml_copysign"
                  [@@unboxed] [@@noalloc]
(** [copysign x y] returns a float whose absolute value is that of [x]
  and whose sign is that of [y].  If [x] is [nan], returns [nan].
  If [y] is [nan], returns either [x] or [-. x], but it is not
  specified which.
  @since 4.00.0  *)

external mod_float : float -> float -> float = "caml_fmod_float" "fmod"
  [@@unboxed] [@@noalloc]
(** [mod_float a b] returns the remainder of [a] with respect to
   [b].  The returned value is [a -. n *. b], where [n]
   is the quotient [a /. b] rounded towards zero to an integer. *)

external frexp : float -> float * int = "caml_frexp_float"
(** [frexp f] returns the pair of the significant
   and the exponent of [f].  When [f] is zero, the
   significant [x] and the exponent [n] of [f] are equal to
   zero.  When [f] is non-zero, they are defined by
   [f = x *. 2 ** n] and [0.5 <= x < 1.0]. *)


external ldexp : (float [@unboxed]) -> (int [@untagged]) -> (float [@unboxed]) =
  "caml_ldexp_float" "caml_ldexp_float_unboxed" [@@noalloc]
(** [ldexp x n] returns [x *. 2 ** n]. *)

external modf : float -> float * float = "caml_modf_float"
(** [modf f] returns the pair of the fractional and integral
   part of [f]. *)

external float : int -> float = "%floatofint"
(** Same as {!Pervasives.float_of_int}. *)

external float_of_int : int -> float = "%floatofint"
(** Convert an integer to floating-point. *)

external truncate : float -> int = "%intoffloat"
(** Same as {!Pervasives.int_of_float}. *)

external int_of_float : float -> int = "%intoffloat"
(** Truncate the given floating-point number to an integer.
   The result is unspecified if the argument is [nan] or falls outside the
   range of representable integers. *)

val infinity : float
(** Positive infinity. *)

val neg_infinity : float
(** Negative infinity. *)

val nan : float
(** A special floating-point value denoting the result of an
   undefined operation such as [0.0 /. 0.0].  Stands for
   'not a number'.  Any floating-point operation with [nan] as
   argument returns [nan] as result.  As for floating-point comparisons,
   [=], [<], [<=], [>] and [>=] return [false] and [<>] returns [true]
   if one or both of their arguments is [nan]. *)

val max_float : float
(** The largest positive finite value of type [float]. *)

val min_float : float
(** The smallest positive, non-zero, non-denormalized value of type [float]. *)

val epsilon_float : float
(** The difference between [1.0] and the smallest exactly representable
    floating-point number greater than [1.0]. *)

type fpclass =
    FP_normal           (** Normal number, none of the below *)
  | FP_subnormal        (** Number very close to 0.0, has reduced precision *)
  | FP_zero             (** Number is 0.0 or -0.0 *)
  | FP_infinite         (** Number is positive or negative infinity *)
  | FP_nan              (** Not a number: result of an undefined operation *)
(** The five classes of floating-point numbers, as determined by
   the {!Pervasives.classify_float} function. *)

external classify_float : (float [@unboxed]) -> fpclass =
  "caml_classify_float" "caml_classify_float_unboxed" [@@noalloc]
(** Return the class of the given floating-point number:
   normal, subnormal, zero, infinite, or not a number. *)


(** {6 String operations}

   More string operations are provided in module {!String}.
*)

val ( ^ ) : string -> string -> string
(** String concatenation. *)


(** {6 Character operations}

   More character operations are provided in module {!Char}.
*)

external int_of_char : char -> int = "%identity"
(** Return the ASCII code of the argument. *)

val char_of_int : int -> char
(** Return the character with the given ASCII code.
   Raise [Invalid_argument "char_of_int"] if the argument is
   outside the range 0--255. *)


(** {6 Unit operations} *)

external ignore : 'a -> unit = "%ignore"
(** Discard the value of its argument and return [()].
   For instance, [ignore(f x)] discards the result of
   the side-effecting function [f].  It is equivalent to
   [f x; ()], except that the latter may generate a
   compiler warning; writing [ignore(f x)] instead
   avoids the warning. *)


(** {6 String conversion functions} *)

val string_of_bool : bool -> string
(** Return the string representation of a boolean. As the returned values
   may be shared, the user should not modify them directly.
*)

val bool_of_string : string -> bool
(** Convert the given string to a boolean.
   Raise [Invalid_argument "bool_of_string"] if the string is not
   ["true"] or ["false"]. *)

val string_of_int : int -> string
(** Return the string representation of an integer, in decimal. *)

external int_of_string : string -> int = "caml_int_of_string"
(** Convert the given string to an integer.
   The string is read in decimal (by default), in hexadecimal (if it
   begins with [0x] or [0X]), in octal (if it begins with [0o] or [0O]),
   or in binary (if it begins with [0b] or [0B]).
   The [_] (underscore) character can appear anywhere in the string
   and is ignored.
   Raise [Failure "int_of_string"] if the given string is not
   a valid representation of an integer, or if the integer represented
   exceeds the range of integers representable in type [int]. *)

val string_of_float : float -> string
(** Return the string representation of a floating-point number. *)

external float_of_string : string -> float = "caml_float_of_string"
(** Convert the given string to a float.  The string is read in decimal
   (by default) or in hexadecimal (marked by [0x] or [0X]).
   The format of decimal floating-point numbers is
   [ [-] dd.ddd (e|E) [+|-] dd ], where [d] stands for a decimal digit.
   The format of hexadecimal floating-point numbers is
   [ [-] 0(x|X) hh.hhh (p|P) [+|-] dd ], where [h] stands for an
   hexadecimal digit and [d] for a decimal digit.
   In both cases, at least one of the integer and fractional parts must be
   given; the exponent part is optional.
   The [_] (underscore) character can appear anywhere in the string
   and is ignored.
   Depending on the execution platforms, other representations of
   floating-point numbers can be accepted, but should not be relied upon.
   Raise [Failure "float_of_string"] if the given string is not a valid
   representation of a float. *)

(** {6 Pair operations} *)

external fst : 'a * 'b -> 'a = "%field0"
(** Return the first component of a pair. *)

external snd : 'a * 'b -> 'b = "%field1"
(** Return the second component of a pair. *)


(** {6 List operations}

   More list operations are provided in module {!List}.
*)

val ( @ ) : 'a list -> 'a list -> 'a list
(** List concatenation.  Not tail-recursive (length of the first argument). *)


(** {6 References} *)

type 'a ref = { mutable contents : 'a }
(** The type of references (mutable indirection cells) containing
   a value of type ['a]. *)

external ref : 'a -> 'a ref = "%makemutable"
(** Return a fresh reference containing the given value. *)

external ( ! ) : 'a ref -> 'a = "%field0"
(** [!r] returns the current contents of reference [r].
   Equivalent to [fun r -> r.contents]. *)

external ( := ) : 'a ref -> 'a -> unit = "%setfield0"
(** [r := a] stores the value of [a] in reference [r].
   Equivalent to [fun r v -> r.contents <- v]. *)

external incr : int ref -> unit = "%incr"
(** Increment the integer contained in the given reference.
   Equivalent to [fun r -> r := succ !r]. *)

external decr : int ref -> unit = "%decr"
(** Decrement the integer contained in the given reference.
   Equivalent to [fun r -> r := pred !r]. *)

(** {6 Result type} *)

type ('a,'b) result = Ok of 'a | Error of 'b

(** {6 Operations on format strings} *)

(** Format strings are character strings with special lexical conventions
  that defines the functionality of formatted input/output functions. Format
  strings are used to read data with formatted input functions from module
  {!Scanf} and to print data with formatted output functions from modules
  {!Printf} and {!Format}.

  Format strings are made of three kinds of entities:
  - {e conversions specifications}, introduced by the special character ['%']
    followed by one or more characters specifying what kind of argument to
    read or print,
  - {e formatting indications}, introduced by the special character ['@']
    followed by one or more characters specifying how to read or print the
    argument,
  - {e plain characters} that are regular characters with usual lexical
    conventions. Plain characters specify string literals to be read in the
    input or printed in the output.

  There is an additional lexical rule to escape the special characters ['%']
  and ['@'] in format strings: if a special character follows a ['%']
  character, it is treated as a plain character. In other words, ["%%"] is
  considered as a plain ['%'] and ["%@"] as a plain ['@'].

  For more information about conversion specifications and formatting
  indications available, read the documentation of modules {!Scanf},
  {!Printf} and {!Format}.
*)

(** Format strings have a general and highly polymorphic type
    [('a, 'b, 'c, 'd, 'e, 'f) format6].
    The two simplified types, [format] and [format4] below are
    included for backward compatibility with earlier releases of
    OCaml.

    The meaning of format string type parameters is as follows:

    - ['a] is the type of the parameters of the format for formatted output
      functions ([printf]-style functions);
      ['a] is the type of the values read by the format for formatted input
      functions ([scanf]-style functions).

    - ['b] is the type of input source for formatted input functions and the
      type of output target for formatted output functions.
      For [printf]-style functions from module [Printf], ['b] is typically
      [out_channel];
      for [printf]-style functions from module [Format], ['b] is typically
      [Format.formatter];
      for [scanf]-style functions from module [Scanf], ['b] is typically
      [Scanf.Scanning.in_channel].

      Type argument ['b] is also the type of the first argument given to
      user's defined printing functions for [%a] and [%t] conversions,
      and user's defined reading functions for [%r] conversion.

    - ['c] is the type of the result of the [%a] and [%t] printing
      functions, and also the type of the argument transmitted to the
      first argument of [kprintf]-style functions or to the
      [kscanf]-style functions.

    - ['d] is the type of parameters for the [scanf]-style functions.

    - ['e] is the type of the receiver function for the [scanf]-style functions.

    - ['f] is the final result type of a formatted input/output function
      invocation: for the [printf]-style functions, it is typically [unit];
      for the [scanf]-style functions, it is typically the result type of the
      receiver function.
*)

type ('a, 'b, 'c, 'd, 'e, 'f) format6 =
  ('a, 'b, 'c, 'd, 'e, 'f) CamlinternalFormatBasics.format6

type ('a, 'b, 'c, 'd) format4 = ('a, 'b, 'c, 'c, 'c, 'd) format6

type ('a, 'b, 'c) format = ('a, 'b, 'c, 'c) format4

val string_of_format : ('a, 'b, 'c, 'd, 'e, 'f) format6 -> string
(** Converts a format string into a string. *)

external format_of_string :
  ('a, 'b, 'c, 'd, 'e, 'f) format6 ->
  ('a, 'b, 'c, 'd, 'e, 'f) format6 = "%identity"
(** [format_of_string s] returns a format string read from the string
    literal [s].
    Note: [format_of_string] can not convert a string argument that is not a
    literal. If you need this functionality, use the more general
    {!Scanf.format_from_string} function.
*)

val ( ^^ ) :
  ('a, 'b, 'c, 'd, 'e, 'f) format6 ->
  ('f, 'b, 'c, 'e, 'g, 'h) format6 ->
  ('a, 'b, 'c, 'd, 'g, 'h) format6
(** [f1 ^^ f2] catenates format strings [f1] and [f2]. The result is a
  format string that behaves as the concatenation of format strings [f1] and
  [f2]: in case of formatted output, it accepts arguments from [f1], then
  arguments from [f2]; in case of formatted input, it returns results from
  [f1], then results from [f2].
*)

end
open Pervasives
module Compare : sig
# 1 "environment/v1/compare.mli"

module type S = sig
  type t
  val (=) : t -> t -> bool
  val (<>) : t -> t -> bool
  val (<) : t -> t -> bool
  val (<=) : t -> t -> bool
  val (>=) : t -> t -> bool
  val (>) : t -> t -> bool
  val compare : t -> t -> int
  val max : t -> t -> t
  val min : t -> t -> t
end

module Char : S with type t = char
module Bool : S with type t = bool
module Int : S with type t = int
module Int32 : S with type t = int32
module Uint32 : S with type t = int32
module Int64 : S with type t = int64
module Uint64 : S with type t = int64
module Float : S with type t = float
module String : S with type t = string
module List(P : S) : S with type t = P.t list
module Option(P : S) : S with type t = P.t option
end
module Array : sig
# 1 "environment/v1/array.mli"
(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*             Xavier Leroy, projet Cristal, INRIA Rocquencourt           *)
(*                                                                        *)
(*   Copyright 1996 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* TEZOS CHANGES

   * Import version 4.04.0
   * Remove unsafe functions
   * Remove deprecated functions

*)

(** Array operations. *)

external length : 'a array -> int = "%array_length"
(** Return the length (number of elements) of the given array. *)

external get : 'a array -> int -> 'a = "%array_safe_get"
(** [Array.get a n] returns the element number [n] of array [a].
   The first element has number 0.
   The last element has number [Array.length a - 1].
   You can also write [a.(n)] instead of [Array.get a n].

   Raise [Invalid_argument "index out of bounds"]
   if [n] is outside the range 0 to [(Array.length a - 1)]. *)

external set : 'a array -> int -> 'a -> unit = "%array_safe_set"
(** [Array.set a n x] modifies array [a] in place, replacing
   element number [n] with [x].
   You can also write [a.(n) <- x] instead of [Array.set a n x].

   Raise [Invalid_argument "index out of bounds"]
   if [n] is outside the range 0 to [Array.length a - 1]. *)

external make : int -> 'a -> 'a array = "caml_make_vect"
(** [Array.make n x] returns a fresh array of length [n],
   initialized with [x].
   All the elements of this new array are initially
   physically equal to [x] (in the sense of the [==] predicate).
   Consequently, if [x] is mutable, it is shared among all elements
   of the array, and modifying [x] through one of the array entries
   will modify all other entries at the same time.

   Raise [Invalid_argument] if [n < 0] or [n > Sys.max_array_length].
   If the value of [x] is a floating-point number, then the maximum
   size is only [Sys.max_array_length / 2].*)

external create_float: int -> float array = "caml_make_float_vect"
(** [Array.create_float n] returns a fresh float array of length [n],
    with uninitialized data.
    @since 4.03 *)

val init : int -> (int -> 'a) -> 'a array
(** [Array.init n f] returns a fresh array of length [n],
   with element number [i] initialized to the result of [f i].
   In other terms, [Array.init n f] tabulates the results of [f]
   applied to the integers [0] to [n-1].

   Raise [Invalid_argument] if [n < 0] or [n > Sys.max_array_length].
   If the return type of [f] is [float], then the maximum
   size is only [Sys.max_array_length / 2].*)

val make_matrix : int -> int -> 'a -> 'a array array
(** [Array.make_matrix dimx dimy e] returns a two-dimensional array
   (an array of arrays) with first dimension [dimx] and
   second dimension [dimy]. All the elements of this new matrix
   are initially physically equal to [e].
   The element ([x,y]) of a matrix [m] is accessed
   with the notation [m.(x).(y)].

   Raise [Invalid_argument] if [dimx] or [dimy] is negative or
   greater than [Sys.max_array_length].
   If the value of [e] is a floating-point number, then the maximum
   size is only [Sys.max_array_length / 2]. *)

val append : 'a array -> 'a array -> 'a array
(** [Array.append v1 v2] returns a fresh array containing the
   concatenation of the arrays [v1] and [v2]. *)

val concat : 'a array list -> 'a array
(** Same as [Array.append], but concatenates a list of arrays. *)

val sub : 'a array -> int -> int -> 'a array
(** [Array.sub a start len] returns a fresh array of length [len],
   containing the elements number [start] to [start + len - 1]
   of array [a].

   Raise [Invalid_argument "Array.sub"] if [start] and [len] do not
   designate a valid subarray of [a]; that is, if
   [start < 0], or [len < 0], or [start + len > Array.length a]. *)

val copy : 'a array -> 'a array
(** [Array.copy a] returns a copy of [a], that is, a fresh array
   containing the same elements as [a]. *)

val fill : 'a array -> int -> int -> 'a -> unit
(** [Array.fill a ofs len x] modifies the array [a] in place,
   storing [x] in elements number [ofs] to [ofs + len - 1].

   Raise [Invalid_argument "Array.fill"] if [ofs] and [len] do not
   designate a valid subarray of [a]. *)

val blit : 'a array -> int -> 'a array -> int -> int -> unit
(** [Array.blit v1 o1 v2 o2 len] copies [len] elements
   from array [v1], starting at element number [o1], to array [v2],
   starting at element number [o2]. It works correctly even if
   [v1] and [v2] are the same array, and the source and
   destination chunks overlap.

   Raise [Invalid_argument "Array.blit"] if [o1] and [len] do not
   designate a valid subarray of [v1], or if [o2] and [len] do not
   designate a valid subarray of [v2]. *)

val to_list : 'a array -> 'a list
(** [Array.to_list a] returns the list of all the elements of [a]. *)

val of_list : 'a list -> 'a array
(** [Array.of_list l] returns a fresh array containing the elements
   of [l]. *)


(** {6 Iterators} *)


val iter : ('a -> unit) -> 'a array -> unit
(** [Array.iter f a] applies function [f] in turn to all
   the elements of [a].  It is equivalent to
   [f a.(0); f a.(1); ...; f a.(Array.length a - 1); ()]. *)

val iteri : (int -> 'a -> unit) -> 'a array -> unit
(** Same as {!Array.iter}, but the
   function is applied with the index of the element as first argument,
   and the element itself as second argument. *)

val map : ('a -> 'b) -> 'a array -> 'b array
(** [Array.map f a] applies function [f] to all the elements of [a],
   and builds an array with the results returned by [f]:
   [[| f a.(0); f a.(1); ...; f a.(Array.length a - 1) |]]. *)

val mapi : (int -> 'a -> 'b) -> 'a array -> 'b array
(** Same as {!Array.map}, but the
   function is applied to the index of the element as first argument,
   and the element itself as second argument. *)

val fold_left : ('a -> 'b -> 'a) -> 'a -> 'b array -> 'a
(** [Array.fold_left f x a] computes
   [f (... (f (f x a.(0)) a.(1)) ...) a.(n-1)],
   where [n] is the length of the array [a]. *)

val fold_right : ('b -> 'a -> 'a) -> 'b array -> 'a -> 'a
(** [Array.fold_right f a x] computes
   [f a.(0) (f a.(1) ( ... (f a.(n-1) x) ...))],
   where [n] is the length of the array [a]. *)


(** {6 Iterators on two arrays} *)


val iter2 : ('a -> 'b -> unit) -> 'a array -> 'b array -> unit
(** [Array.iter2 f a b] applies function [f] to all the elements of [a]
   and [b].
   Raise [Invalid_argument] if the arrays are not the same size.
   @since 4.03.0 *)

val map2 : ('a -> 'b -> 'c) -> 'a array -> 'b array -> 'c array
(** [Array.map2 f a b] applies function [f] to all the elements of [a]
   and [b], and builds an array with the results returned by [f]:
   [[| f a.(0) b.(0); ...; f a.(Array.length a - 1) b.(Array.length b - 1)|]].
   Raise [Invalid_argument] if the arrays are not the same size.
   @since 4.03.0 *)


(** {6 Array scanning} *)


val for_all : ('a -> bool) -> 'a array -> bool
(** [Array.for_all p [|a1; ...; an|]] checks if all elements of the array
   satisfy the predicate [p]. That is, it returns
   [(p a1) && (p a2) && ... && (p an)].
   @since 4.03.0 *)

val exists : ('a -> bool) -> 'a array -> bool
(** [Array.exists p [|a1; ...; an|]] checks if at least one element of
    the array satisfies the predicate [p]. That is, it returns
    [(p a1) || (p a2) || ... || (p an)].
    @since 4.03.0 *)

val mem : 'a -> 'a array -> bool
(** [mem a l] is true if and only if [a] is equal
   to an element of [l].
   @since 4.03.0 *)

val memq : 'a -> 'a array -> bool
(** Same as {!Array.mem}, but uses physical equality instead of structural
   equality to compare array elements.
   @since 4.03.0 *)


(** {6 Sorting} *)


val sort : ('a -> 'a -> int) -> 'a array -> unit
(** Sort an array in increasing order according to a comparison
   function.  The comparison function must return 0 if its arguments
   compare as equal, a positive integer if the first is greater,
   and a negative integer if the first is smaller (see below for a
   complete specification).  For example, {!Pervasives.compare} is
   a suitable comparison function, provided there are no floating-point
   NaN values in the data.  After calling [Array.sort], the
   array is sorted in place in increasing order.
   [Array.sort] is guaranteed to run in constant heap space
   and (at most) logarithmic stack space.

   The current implementation uses Heap Sort.  It runs in constant
   stack space.

   Specification of the comparison function:
   Let [a] be the array and [cmp] the comparison function.  The following
   must be true for all x, y, z in a :
-   [cmp x y] > 0 if and only if [cmp y x] < 0
-   if [cmp x y] >= 0 and [cmp y z] >= 0 then [cmp x z] >= 0

   When [Array.sort] returns, [a] contains the same elements as before,
   reordered in such a way that for all i and j valid indices of [a] :
-   [cmp a.(i) a.(j)] >= 0 if and only if i >= j
*)

val stable_sort : ('a -> 'a -> int) -> 'a array -> unit
(** Same as {!Array.sort}, but the sorting algorithm is stable (i.e.
   elements that compare equal are kept in their original order) and
   not guaranteed to run in constant heap space.

   The current implementation uses Merge Sort. It uses [n/2]
   words of heap space, where [n] is the length of the array.
   It is usually faster than the current implementation of {!Array.sort}.
*)

val fast_sort : ('a -> 'a -> int) -> 'a array -> unit
(** Same as {!Array.sort} or {!Array.stable_sort}, whichever is faster
    on typical input.
*)

end
module List : sig
# 1 "environment/v1/list.mli"
(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*             Xavier Leroy, projet Cristal, INRIA Rocquencourt           *)
(*                                                                        *)
(*   Copyright 1996 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(** List operations.

   Some functions are flagged as not tail-recursive.  A tail-recursive
   function uses constant stack space, while a non-tail-recursive function
   uses stack space proportional to the length of its list argument, which
   can be a problem with very long lists.  When the function takes several
   list arguments, an approximate formula giving stack usage (in some
   unspecified constant unit) is shown in parentheses.

   The above considerations can usually be ignored if your lists are not
   longer than about 10000 elements.
*)

val length : 'a list -> int
(** Return the length (number of elements) of the given list. *)

val cons : 'a -> 'a list -> 'a list
(** [cons x xs] is [x :: xs]
    @since 4.03.0
*)

val hd : 'a list -> 'a
(** Return the first element of the given list. Raise
   [Failure "hd"] if the list is empty. *)

val tl : 'a list -> 'a list
(** Return the given list without its first element. Raise
   [Failure "tl"] if the list is empty. *)

val nth : 'a list -> int -> 'a
(** Return the [n]-th element of the given list.
   The first element (head of the list) is at position 0.
   Raise [Failure "nth"] if the list is too short.
   Raise [Invalid_argument "List.nth"] if [n] is negative. *)

val rev : 'a list -> 'a list
(** List reversal. *)

val append : 'a list -> 'a list -> 'a list
(** Concatenate two lists.  Same as the infix operator [@].
   Not tail-recursive (length of the first argument).  *)

val rev_append : 'a list -> 'a list -> 'a list
(** [List.rev_append l1 l2] reverses [l1] and concatenates it to [l2].
   This is equivalent to {!List.rev}[ l1 @ l2], but [rev_append] is
   tail-recursive and more efficient. *)

val concat : 'a list list -> 'a list
(** Concatenate a list of lists.  The elements of the argument are all
   concatenated together (in the same order) to give the result.
   Not tail-recursive
   (length of the argument + length of the longest sub-list). *)

val flatten : 'a list list -> 'a list
(** An alias for [concat]. *)


(** {6 Iterators} *)


val iter : ('a -> unit) -> 'a list -> unit
(** [List.iter f [a1; ...; an]] applies function [f] in turn to
   [a1; ...; an]. It is equivalent to
   [begin f a1; f a2; ...; f an; () end]. *)

val iteri : (int -> 'a -> unit) -> 'a list -> unit
(** Same as {!List.iter}, but the function is applied to the index of
   the element as first argument (counting from 0), and the element
   itself as second argument.
   @since 4.00.0
*)

val map : ('a -> 'b) -> 'a list -> 'b list
(** [List.map f [a1; ...; an]] applies function [f] to [a1, ..., an],
   and builds the list [[f a1; ...; f an]]
   with the results returned by [f].  Not tail-recursive. *)

val mapi : (int -> 'a -> 'b) -> 'a list -> 'b list
(** Same as {!List.map}, but the function is applied to the index of
   the element as first argument (counting from 0), and the element
   itself as second argument.  Not tail-recursive.
   @since 4.00.0
*)

val rev_map : ('a -> 'b) -> 'a list -> 'b list
(** [List.rev_map f l] gives the same result as
   {!List.rev}[ (]{!List.map}[ f l)], but is tail-recursive and
   more efficient. *)

val fold_left : ('a -> 'b -> 'a) -> 'a -> 'b list -> 'a
(** [List.fold_left f a [b1; ...; bn]] is
   [f (... (f (f a b1) b2) ...) bn]. *)

val fold_right : ('a -> 'b -> 'b) -> 'a list -> 'b -> 'b
(** [List.fold_right f [a1; ...; an] b] is
   [f a1 (f a2 (... (f an b) ...))].  Not tail-recursive. *)


(** {6 Iterators on two lists} *)


val iter2 : ('a -> 'b -> unit) -> 'a list -> 'b list -> unit
(** [List.iter2 f [a1; ...; an] [b1; ...; bn]] calls in turn
   [f a1 b1; ...; f an bn].
   Raise [Invalid_argument] if the two lists are determined
   to have different lengths. *)

val map2 : ('a -> 'b -> 'c) -> 'a list -> 'b list -> 'c list
(** [List.map2 f [a1; ...; an] [b1; ...; bn]] is
   [[f a1 b1; ...; f an bn]].
   Raise [Invalid_argument] if the two lists are determined
   to have different lengths.  Not tail-recursive. *)

val rev_map2 : ('a -> 'b -> 'c) -> 'a list -> 'b list -> 'c list
(** [List.rev_map2 f l1 l2] gives the same result as
   {!List.rev}[ (]{!List.map2}[ f l1 l2)], but is tail-recursive and
   more efficient. *)

val fold_left2 : ('a -> 'b -> 'c -> 'a) -> 'a -> 'b list -> 'c list -> 'a
(** [List.fold_left2 f a [b1; ...; bn] [c1; ...; cn]] is
   [f (... (f (f a b1 c1) b2 c2) ...) bn cn].
   Raise [Invalid_argument] if the two lists are determined
   to have different lengths. *)

val fold_right2 : ('a -> 'b -> 'c -> 'c) -> 'a list -> 'b list -> 'c -> 'c
(** [List.fold_right2 f [a1; ...; an] [b1; ...; bn] c] is
   [f a1 b1 (f a2 b2 (... (f an bn c) ...))].
   Raise [Invalid_argument] if the two lists are determined
   to have different lengths.  Not tail-recursive. *)


(** {6 List scanning} *)


val for_all : ('a -> bool) -> 'a list -> bool
(** [for_all p [a1; ...; an]] checks if all elements of the list
   satisfy the predicate [p]. That is, it returns
   [(p a1) && (p a2) && ... && (p an)]. *)

val exists : ('a -> bool) -> 'a list -> bool
(** [exists p [a1; ...; an]] checks if at least one element of
   the list satisfies the predicate [p]. That is, it returns
   [(p a1) || (p a2) || ... || (p an)]. *)

val for_all2 : ('a -> 'b -> bool) -> 'a list -> 'b list -> bool
(** Same as {!List.for_all}, but for a two-argument predicate.
   Raise [Invalid_argument] if the two lists are determined
   to have different lengths. *)

val exists2 : ('a -> 'b -> bool) -> 'a list -> 'b list -> bool
(** Same as {!List.exists}, but for a two-argument predicate.
   Raise [Invalid_argument] if the two lists are determined
   to have different lengths. *)

val mem : 'a -> 'a list -> bool
(** [mem a l] is true if and only if [a] is equal
   to an element of [l]. *)

val memq : 'a -> 'a list -> bool
(** Same as {!List.mem}, but uses physical equality instead of structural
   equality to compare list elements. *)


(** {6 List searching} *)


val find : ('a -> bool) -> 'a list -> 'a
(** [find p l] returns the first element of the list [l]
   that satisfies the predicate [p].
   Raise [Not_found] if there is no value that satisfies [p] in the
   list [l]. *)

val filter : ('a -> bool) -> 'a list -> 'a list
(** [filter p l] returns all the elements of the list [l]
   that satisfy the predicate [p].  The order of the elements
   in the input list is preserved.  *)

val find_all : ('a -> bool) -> 'a list -> 'a list
(** [find_all] is another name for {!List.filter}. *)

val partition : ('a -> bool) -> 'a list -> 'a list * 'a list
(** [partition p l] returns a pair of lists [(l1, l2)], where
   [l1] is the list of all the elements of [l] that
   satisfy the predicate [p], and [l2] is the list of all the
   elements of [l] that do not satisfy [p].
   The order of the elements in the input list is preserved. *)


(** {6 Association lists} *)


val assoc : 'a -> ('a * 'b) list -> 'b
(** [assoc a l] returns the value associated with key [a] in the list of
   pairs [l]. That is,
   [assoc a [ ...; (a,b); ...] = b]
   if [(a,b)] is the leftmost binding of [a] in list [l].
   Raise [Not_found] if there is no value associated with [a] in the
   list [l]. *)

val assq : 'a -> ('a * 'b) list -> 'b
(** Same as {!List.assoc}, but uses physical equality instead of structural
   equality to compare keys. *)

val mem_assoc : 'a -> ('a * 'b) list -> bool
(** Same as {!List.assoc}, but simply return true if a binding exists,
   and false if no bindings exist for the given key. *)

val mem_assq : 'a -> ('a * 'b) list -> bool
(** Same as {!List.mem_assoc}, but uses physical equality instead of
   structural equality to compare keys. *)

val remove_assoc : 'a -> ('a * 'b) list -> ('a * 'b) list
(** [remove_assoc a l] returns the list of
   pairs [l] without the first pair with key [a], if any.
   Not tail-recursive. *)

val remove_assq : 'a -> ('a * 'b) list -> ('a * 'b) list
(** Same as {!List.remove_assoc}, but uses physical equality instead
   of structural equality to compare keys.  Not tail-recursive. *)


(** {6 Lists of pairs} *)


val split : ('a * 'b) list -> 'a list * 'b list
(** Transform a list of pairs into a pair of lists:
   [split [(a1,b1); ...; (an,bn)]] is [([a1; ...; an], [b1; ...; bn])].
   Not tail-recursive.
*)

val combine : 'a list -> 'b list -> ('a * 'b) list
(** Transform a pair of lists into a list of pairs:
   [combine [a1; ...; an] [b1; ...; bn]] is
   [[(a1,b1); ...; (an,bn)]].
   Raise [Invalid_argument] if the two lists
   have different lengths.  Not tail-recursive. *)


(** {6 Sorting} *)


val sort : ('a -> 'a -> int) -> 'a list -> 'a list
(** Sort a list in increasing order according to a comparison
   function.  The comparison function must return 0 if its arguments
   compare as equal, a positive integer if the first is greater,
   and a negative integer if the first is smaller (see Array.sort for
   a complete specification).  For example,
   {!Pervasives.compare} is a suitable comparison function.
   The resulting list is sorted in increasing order.
   [List.sort] is guaranteed to run in constant heap space
   (in addition to the size of the result list) and logarithmic
   stack space.

   The current implementation uses Merge Sort. It runs in constant
   heap space and logarithmic stack space.
*)

val stable_sort : ('a -> 'a -> int) -> 'a list -> 'a list
(** Same as {!List.sort}, but the sorting algorithm is guaranteed to
   be stable (i.e. elements that compare equal are kept in their
   original order) .

   The current implementation uses Merge Sort. It runs in constant
   heap space and logarithmic stack space.
*)

val fast_sort : ('a -> 'a -> int) -> 'a list -> 'a list
(** Same as {!List.sort} or {!List.stable_sort}, whichever is faster
    on typical input. *)

val sort_uniq : ('a -> 'a -> int) -> 'a list -> 'a list
(** Same as {!List.sort}, but also remove duplicates.
    @since 4.02.0 *)

val merge : ('a -> 'a -> int) -> 'a list -> 'a list -> 'a list
(** Merge two lists:
    Assuming that [l1] and [l2] are sorted according to the
    comparison function [cmp], [merge cmp l1 l2] will return a
    sorted list containting all the elements of [l1] and [l2].
    If several elements compare equal, the elements of [l1] will be
    before the elements of [l2].
    Not tail-recursive (sum of the lengths of the arguments).
*)
end
module Bytes : sig
# 1 "environment/v1/bytes.mli"
(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*             Xavier Leroy, projet Cristal, INRIA Rocquencourt           *)
(*                                                                        *)
(*   Copyright 1996 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* TEZOS CHANGES

   * Import version 4.04.0
   * Remove unsafe functions
   * Remove deprecated functions
   * Add binary data insertion / extraction functions

*)

(** Byte sequence operations.

   A byte sequence is a mutable data structure that contains a
   fixed-length sequence of bytes. Each byte can be indexed in
   constant time for reading or writing.

   Given a byte sequence [s] of length [l], we can access each of the
   [l] bytes of [s] via its index in the sequence. Indexes start at
   [0], and we will call an index valid in [s] if it falls within the
   range [[0...l-1]] (inclusive). A position is the point between two
   bytes or at the beginning or end of the sequence.  We call a
   position valid in [s] if it falls within the range [[0...l]]
   (inclusive). Note that the byte at index [n] is between positions
   [n] and [n+1].

   Two parameters [start] and [len] are said to designate a valid
   range of [s] if [len >= 0] and [start] and [start+len] are valid
   positions in [s].

   Byte sequences can be modified in place, for instance via the [set]
   and [blit] functions described below.  See also strings (module
   {!String}), which are almost the same data structure, but cannot be
   modified in place.

   Bytes are represented by the OCaml type [char].

   @since 4.02.0
 *)

external length : bytes -> int = "%bytes_length"
(** Return the length (number of bytes) of the argument. *)

external get : bytes -> int -> char = "%bytes_safe_get"
(** [get s n] returns the byte at index [n] in argument [s].

    Raise [Invalid_argument] if [n] not a valid index in [s]. *)

external set : bytes -> int -> char -> unit = "%bytes_safe_set"
(** [set s n c] modifies [s] in place, replacing the byte at index [n]
    with [c].

    Raise [Invalid_argument] if [n] is not a valid index in [s]. *)

external create : int -> bytes = "caml_create_bytes"
(** [create n] returns a new byte sequence of length [n]. The
    sequence is uninitialized and contains arbitrary bytes.

    Raise [Invalid_argument] if [n < 0] or [n > ]{!Sys.max_string_length}. *)

val make : int -> char -> bytes
(** [make n c] returns a new byte sequence of length [n], filled with
    the byte [c].

    Raise [Invalid_argument] if [n < 0] or [n > ]{!Sys.max_string_length}. *)

val init : int -> (int -> char) -> bytes
(** [Bytes.init n f] returns a fresh byte sequence of length [n], with
    character [i] initialized to the result of [f i] (in increasing
    index order).

    Raise [Invalid_argument] if [n < 0] or [n > ]{!Sys.max_string_length}. *)

val empty : bytes
(** A byte sequence of size 0. *)

val copy : bytes -> bytes
(** Return a new byte sequence that contains the same bytes as the
    argument. *)

val of_string : string -> bytes
(** Return a new byte sequence that contains the same bytes as the
    given string. *)

val to_string : bytes -> string
(** Return a new string that contains the same bytes as the given byte
    sequence. *)

val sub : bytes -> int -> int -> bytes
(** [sub s start len] returns a new byte sequence of length [len],
    containing the subsequence of [s] that starts at position [start]
    and has length [len].

    Raise [Invalid_argument] if [start] and [len] do not designate a
    valid range of [s]. *)

val sub_string : bytes -> int -> int -> string
(** Same as [sub] but return a string instead of a byte sequence. *)

val extend : bytes -> int -> int -> bytes
(** [extend s left right] returns a new byte sequence that contains
    the bytes of [s], with [left] uninitialized bytes prepended and
    [right] uninitialized bytes appended to it. If [left] or [right]
    is negative, then bytes are removed (instead of appended) from
    the corresponding side of [s].

    Raise [Invalid_argument] if the result length is negative or
    longer than {!Sys.max_string_length} bytes. *)

val fill : bytes -> int -> int -> char -> unit
(** [fill s start len c] modifies [s] in place, replacing [len]
    characters with [c], starting at [start].

    Raise [Invalid_argument] if [start] and [len] do not designate a
    valid range of [s]. *)

val blit : bytes -> int -> bytes -> int -> int -> unit
(** [blit src srcoff dst dstoff len] copies [len] bytes from sequence
    [src], starting at index [srcoff], to sequence [dst], starting at
    index [dstoff]. It works correctly even if [src] and [dst] are the
    same byte sequence, and the source and destination intervals
    overlap.

    Raise [Invalid_argument] if [srcoff] and [len] do not
    designate a valid range of [src], or if [dstoff] and [len]
    do not designate a valid range of [dst]. *)

val blit_string : string -> int -> bytes -> int -> int -> unit
(** [blit src srcoff dst dstoff len] copies [len] bytes from string
    [src], starting at index [srcoff], to byte sequence [dst],
    starting at index [dstoff].

    Raise [Invalid_argument] if [srcoff] and [len] do not
    designate a valid range of [src], or if [dstoff] and [len]
    do not designate a valid range of [dst]. *)

val concat : bytes -> bytes list -> bytes
(** [concat sep sl] concatenates the list of byte sequences [sl],
    inserting the separator byte sequence [sep] between each, and
    returns the result as a new byte sequence.

    Raise [Invalid_argument] if the result is longer than
    {!Sys.max_string_length} bytes. *)

val cat : bytes -> bytes -> bytes
(** [cat s1 s2] concatenates [s1] and [s2] and returns the result
     as new byte sequence.

    Raise [Invalid_argument] if the result is longer than
    {!Sys.max_string_length} bytes. *)

val iter : (char -> unit) -> bytes -> unit
(** [iter f s] applies function [f] in turn to all the bytes of [s].
    It is equivalent to [f (get s 0); f (get s 1); ...; f (get s
    (length s - 1)); ()]. *)

val iteri : (int -> char -> unit) -> bytes -> unit
(** Same as {!Bytes.iter}, but the function is applied to the index of
    the byte as first argument and the byte itself as second
    argument. *)

val map : (char -> char) -> bytes -> bytes
(** [map f s] applies function [f] in turn to all the bytes of [s]
    (in increasing index order) and stores the resulting bytes in
    a new sequence that is returned as the result. *)

val mapi : (int -> char -> char) -> bytes -> bytes
(** [mapi f s] calls [f] with each character of [s] and its
    index (in increasing index order) and stores the resulting bytes
    in a new sequence that is returned as the result. *)

val trim : bytes -> bytes
(** Return a copy of the argument, without leading and trailing
    whitespace. The bytes regarded as whitespace are the ASCII
    characters [' '], ['\012'], ['\n'], ['\r'], and ['\t']. *)

val escaped : bytes -> bytes
(** Return a copy of the argument, with special characters represented
    by escape sequences, following the lexical conventions of OCaml.
    All characters outside the ASCII printable range (32..126) are
    escaped, as well as backslash and double-quote.

    Raise [Invalid_argument] if the result is longer than
    {!Sys.max_string_length} bytes. *)

val index : bytes -> char -> int
(** [index s c] returns the index of the first occurrence of byte [c]
    in [s].

    Raise [Not_found] if [c] does not occur in [s]. *)

val rindex : bytes -> char -> int
(** [rindex s c] returns the index of the last occurrence of byte [c]
    in [s].

    Raise [Not_found] if [c] does not occur in [s]. *)

val index_from : bytes -> int -> char -> int
(** [index_from s i c] returns the index of the first occurrence of
    byte [c] in [s] after position [i].  [Bytes.index s c] is
    equivalent to [Bytes.index_from s 0 c].

    Raise [Invalid_argument] if [i] is not a valid position in [s].
    Raise [Not_found] if [c] does not occur in [s] after position [i]. *)

val rindex_from : bytes -> int -> char -> int
(** [rindex_from s i c] returns the index of the last occurrence of
    byte [c] in [s] before position [i+1].  [rindex s c] is equivalent
    to [rindex_from s (Bytes.length s - 1) c].

    Raise [Invalid_argument] if [i+1] is not a valid position in [s].
    Raise [Not_found] if [c] does not occur in [s] before position [i+1]. *)

val contains : bytes -> char -> bool
(** [contains s c] tests if byte [c] appears in [s]. *)

val contains_from : bytes -> int -> char -> bool
(** [contains_from s start c] tests if byte [c] appears in [s] after
    position [start].  [contains s c] is equivalent to [contains_from
    s 0 c].

    Raise [Invalid_argument] if [start] is not a valid position in [s]. *)

val rcontains_from : bytes -> int -> char -> bool
(** [rcontains_from s stop c] tests if byte [c] appears in [s] before
    position [stop+1].

    Raise [Invalid_argument] if [stop < 0] or [stop+1] is not a valid
    position in [s]. *)

val uppercase_ascii : bytes -> bytes
(** Return a copy of the argument, with all lowercase letters
   translated to uppercase, using the US-ASCII character set.
   @since 4.03.0 *)

val lowercase_ascii : bytes -> bytes
(** Return a copy of the argument, with all uppercase letters
   translated to lowercase, using the US-ASCII character set.
   @since 4.03.0 *)

val capitalize_ascii : bytes -> bytes
(** Return a copy of the argument, with the first character set to uppercase,
   using the US-ASCII character set.
   @since 4.03.0 *)

val uncapitalize_ascii : bytes -> bytes
(** Return a copy of the argument, with the first character set to lowercase,
   using the US-ASCII character set.
   @since 4.03.0 *)

type t = bytes
(** An alias for the type of byte sequences. *)

val compare: t -> t -> int
(** The comparison function for byte sequences, with the same
    specification as {!Pervasives.compare}.  Along with the type [t],
    this function [compare] allows the module [Bytes] to be passed as
    argument to the functors {!Set.Make} and {!Map.Make}. *)

val equal: t -> t -> bool
(** The equality function for byte sequences.
    @since 4.03.0 *)

(** {4 Unsafe conversions (for advanced users)}

    This section describes unsafe, low-level conversion functions
    between [bytes] and [string]. They do not copy the internal data;
    used improperly, they can break the immutability invariant on
    strings provided by the [-safe-string] option. They are available for
    expert library authors, but for most purposes you should use the
    always-correct {!Bytes.to_string} and {!Bytes.of_string} instead.
*)

(** Functions reading and writing bytes  *)

val get_char: t -> int -> char
(** [get_char buff i] reads 1 byte at offset i as a char *)

val get_uint8: t -> int -> int
(** [get_uint8 buff i] reads 1 byte at offset i as an unsigned int of 8
    bits. i.e. It returns a value between 0 and 2^8-1 *)

val get_int8: t -> int -> int
(** [get_int8 buff i] reads 1 byte at offset i as a signed int of 8
    bits. i.e. It returns a value between -2^7 and 2^7-1 *)

val set_char: t -> int -> char -> unit
(** [set_char buff i v] writes [v] to [buff] at offset [i] *)

val set_int8: t -> int -> int -> unit
(** [set_int8 buff i v] writes the least significant 8 bits of [v]
    to [buff] at offset [i] *)

(** Functions reading according to Big Endian byte order *)

val get_uint16: t -> int -> int
(** [get_uint16 buff i] reads 2 bytes at offset i as an unsigned int
      of 16 bits. i.e. It returns a value between 0 and 2^16-1 *)

val get_int16: t -> int -> int
(** [get_int16 buff i] reads 2 byte at offset i as a signed int of
      16 bits. i.e. It returns a value between -2^15 and 2^15-1 *)

val get_int32: t -> int -> int32
(** [get_int32 buff i] reads 4 bytes at offset i as an int32. *)

val get_int64: t -> int -> int64
(** [get_int64 buff i] reads 8 bytes at offset i as an int64. *)

val set_int16: t -> int -> int -> unit
(** [set_int16 buff i v] writes the least significant 16 bits of [v]
      to [buff] at offset [i] *)

val set_int32: t -> int -> int32 -> unit
(** [set_int32 buff i v] writes [v] to [buff] at offset [i] *)

val set_int64: t -> int -> int64 -> unit
(** [set_int64 buff i v] writes [v] to [buff] at offset [i] *)


module LE: sig

  (** Functions reading according to Little Endian byte order *)

  val get_uint16: t -> int -> int
  (** [get_uint16 buff i] reads 2 bytes at offset i as an unsigned int
      of 16 bits. i.e. It returns a value between 0 and 2^16-1 *)

  val get_int16: t -> int -> int
  (** [get_int16 buff i] reads 2 byte at offset i as a signed int of
      16 bits. i.e. It returns a value between -2^15 and 2^15-1 *)

  val get_int32: t -> int -> int32
  (** [get_int32 buff i] reads 4 bytes at offset i as an int32. *)

  val get_int64: t -> int -> int64
  (** [get_int64 buff i] reads 8 bytes at offset i as an int64. *)

  val set_int16: t -> int -> int -> unit
  (** [set_int16 buff i v] writes the least significant 16 bits of [v]
      to [buff] at offset [i] *)

  val set_int32: t -> int -> int32 -> unit
  (** [set_int32 buff i v] writes [v] to [buff] at offset [i] *)

  val set_int64: t -> int -> int64 -> unit
  (** [set_int64 buff i v] writes [v] to [buff] at offset [i] *)

end
end
module String : sig
# 1 "environment/v1/string.mli"
(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*             Xavier Leroy, projet Cristal, INRIA Rocquencourt           *)
(*                                                                        *)
(*   Copyright 1996 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* TEZOS CHANGES

   * Import version 4.04.0
   * Remove unsafe functions
   * Remove deprecated functions (enforcing string immutability)
   * Add binary data extraction functions

*)

(** String operations.

  A string is an immutable data structure that contains a
  fixed-length sequence of (single-byte) characters. Each character
  can be accessed in constant time through its index.

  Given a string [s] of length [l], we can access each of the [l]
  characters of [s] via its index in the sequence. Indexes start at
  [0], and we will call an index valid in [s] if it falls within the
  range [[0...l-1]] (inclusive). A position is the point between two
  characters or at the beginning or end of the string.  We call a
  position valid in [s] if it falls within the range [[0...l]]
  (inclusive). Note that the character at index [n] is between
  positions [n] and [n+1].

  Two parameters [start] and [len] are said to designate a valid
  substring of [s] if [len >= 0] and [start] and [start+len] are
  valid positions in [s].

 *)

external length : string -> int = "%string_length"
(** Return the length (number of characters) of the given string. *)

external get : string -> int -> char = "%string_safe_get"
(** [String.get s n] returns the character at index [n] in string [s].
   You can also write [s.[n]] instead of [String.get s n].

   Raise [Invalid_argument] if [n] not a valid index in [s]. *)


val make : int -> char -> string
(** [String.make n c] returns a fresh string of length [n],
   filled with the character [c].

   Raise [Invalid_argument] if [n < 0] or [n > ]{!Sys.max_string_length}. *)

val init : int -> (int -> char) -> string
(** [String.init n f] returns a string of length [n], with character
    [i] initialized to the result of [f i] (called in increasing
    index order).

    Raise [Invalid_argument] if [n < 0] or [n > ]{!Sys.max_string_length}.

    @since 4.02.0
*)

val sub : string -> int -> int -> string
(** [String.sub s start len] returns a fresh string of length [len],
   containing the substring of [s] that starts at position [start] and
   has length [len].

   Raise [Invalid_argument] if [start] and [len] do not
   designate a valid substring of [s]. *)

val blit : string -> int -> bytes -> int -> int -> unit
(** Same as {!Bytes.blit_string}. *)

val concat : string -> string list -> string
(** [String.concat sep sl] concatenates the list of strings [sl],
    inserting the separator string [sep] between each.

    Raise [Invalid_argument] if the result is longer than
    {!Sys.max_string_length} bytes. *)

val iter : (char -> unit) -> string -> unit
(** [String.iter f s] applies function [f] in turn to all
   the characters of [s].  It is equivalent to
   [f s.[0]; f s.[1]; ...; f s.[String.length s - 1]; ()]. *)

val iteri : (int -> char -> unit) -> string -> unit
(** Same as {!String.iter}, but the
   function is applied to the index of the element as first argument
   (counting from 0), and the character itself as second argument.
   @since 4.00.0 *)

val map : (char -> char) -> string -> string
(** [String.map f s] applies function [f] in turn to all the
    characters of [s] (in increasing index order) and stores the
    results in a new string that is returned.
    @since 4.00.0 *)

val mapi : (int -> char -> char) -> string -> string
(** [String.mapi f s] calls [f] with each character of [s] and its
    index (in increasing index order) and stores the results in a new
    string that is returned.
    @since 4.02.0 *)

val trim : string -> string
(** Return a copy of the argument, without leading and trailing
   whitespace.  The characters regarded as whitespace are: [' '],
   ['\012'], ['\n'], ['\r'], and ['\t'].  If there is neither leading nor
   trailing whitespace character in the argument, return the original
   string itself, not a copy.
   @since 4.00.0 *)

val escaped : string -> string
(** Return a copy of the argument, with special characters
    represented by escape sequences, following the lexical
    conventions of OCaml.
    All characters outside the ASCII printable range (32..126) are
    escaped, as well as backslash and double-quote.

    If there is no special character in the argument that needs
    escaping, return the original string itself, not a copy.

    Raise [Invalid_argument] if the result is longer than
    {!Sys.max_string_length} bytes.

    The function {!Scanf.unescaped} is a left inverse of [escaped],
    i.e. [Scanf.unescaped (escaped s) = s] for any string [s] (unless
    [escape s] fails). *)

val index : string -> char -> int
(** [String.index s c] returns the index of the first
   occurrence of character [c] in string [s].

   Raise [Not_found] if [c] does not occur in [s]. *)

val rindex : string -> char -> int
(** [String.rindex s c] returns the index of the last
   occurrence of character [c] in string [s].

   Raise [Not_found] if [c] does not occur in [s]. *)

val index_from : string -> int -> char -> int
(** [String.index_from s i c] returns the index of the
   first occurrence of character [c] in string [s] after position [i].
   [String.index s c] is equivalent to [String.index_from s 0 c].

   Raise [Invalid_argument] if [i] is not a valid position in [s].
   Raise [Not_found] if [c] does not occur in [s] after position [i]. *)

val rindex_from : string -> int -> char -> int
(** [String.rindex_from s i c] returns the index of the
   last occurrence of character [c] in string [s] before position [i+1].
   [String.rindex s c] is equivalent to
   [String.rindex_from s (String.length s - 1) c].

   Raise [Invalid_argument] if [i+1] is not a valid position in [s].
   Raise [Not_found] if [c] does not occur in [s] before position [i+1]. *)

val contains : string -> char -> bool
(** [String.contains s c] tests if character [c]
   appears in the string [s]. *)

val contains_from : string -> int -> char -> bool
(** [String.contains_from s start c] tests if character [c]
   appears in [s] after position [start].
   [String.contains s c] is equivalent to
   [String.contains_from s 0 c].

   Raise [Invalid_argument] if [start] is not a valid position in [s]. *)

val rcontains_from : string -> int -> char -> bool
(** [String.rcontains_from s stop c] tests if character [c]
   appears in [s] before position [stop+1].

   Raise [Invalid_argument] if [stop < 0] or [stop+1] is not a valid
   position in [s]. *)

val uppercase_ascii : string -> string
(** Return a copy of the argument, with all lowercase letters
   translated to uppercase, using the US-ASCII character set.
   @since 4.03.0 *)

val lowercase_ascii : string -> string
(** Return a copy of the argument, with all uppercase letters
   translated to lowercase, using the US-ASCII character set.
   @since 4.03.0 *)

val capitalize_ascii : string -> string
(** Return a copy of the argument, with the first character set to uppercase,
   using the US-ASCII character set.
   @since 4.03.0 *)

val uncapitalize_ascii : string -> string
(** Return a copy of the argument, with the first character set to lowercase,
   using the US-ASCII character set.
   @since 4.03.0 *)

type t = string
(** An alias for the type of strings. *)

val compare: t -> t -> int
(** The comparison function for strings, with the same specification as
    {!Pervasives.compare}.  Along with the type [t], this function [compare]
    allows the module [String] to be passed as argument to the functors
    {!Set.Make} and {!Map.Make}. *)

val equal: t -> t -> bool
(** The equal function for strings.
    @since 4.03.0 *)

val split_on_char: char -> string -> string list
(** [String.split_on_char sep s] returns the list of all (possibly empty)
    substrings of [s] that are delimited by the [sep] character.

    The function's output is specified by the following invariants:

    - The list is not empty.
    - Concatenating its elements using [sep] as a separator returns a
      string equal to the input ([String.concat (String.make 1 sep)
      (String.split_on_char sep s) = s]).
    - No string in the result contains the [sep] character.

    @since 4.04.0
*)

(** Functions reading bytes  *)

val get_char: t -> int -> char
(** [get_char buff i] reads 1 byte at offset i as a char *)

val get_uint8: t -> int -> int
(** [get_uint8 buff i] reads 1 byte at offset i as an unsigned int of 8
    bits. i.e. It returns a value between 0 and 2^8-1 *)

val get_int8: t -> int -> int
(** [get_int8 buff i] reads 1 byte at offset i as a signed int of 8
    bits. i.e. It returns a value between -2^7 and 2^7-1 *)

(** Functions reading according to Big Endian byte order *)

val get_uint16: t -> int -> int
(** [get_uint16 buff i] reads 2 bytes at offset i as an unsigned int
      of 16 bits. i.e. It returns a value between 0 and 2^16-1 *)

val get_int16: t -> int -> int
(** [get_int16 buff i] reads 2 byte at offset i as a signed int of
      16 bits. i.e. It returns a value between -2^15 and 2^15-1 *)

val get_int32: t -> int -> int32
(** [get_int32 buff i] reads 4 bytes at offset i as an int32. *)

val get_int64: t -> int -> int64
(** [get_int64 buff i] reads 8 bytes at offset i as an int64. *)

module LE: sig

  (** Functions reading according to Little Endian byte order *)

  val get_uint16: t -> int -> int
  (** [get_uint16 buff i] reads 2 bytes at offset i as an unsigned int
      of 16 bits. i.e. It returns a value between 0 and 2^16-1 *)

  val get_int16: t -> int -> int
  (** [get_int16 buff i] reads 2 byte at offset i as a signed int of
      16 bits. i.e. It returns a value between -2^15 and 2^15-1 *)

  val get_int32: t -> int -> int32
  (** [get_int32 buff i] reads 4 bytes at offset i as an int32. *)

  val get_int64: t -> int -> int64
  (** [get_int64 buff i] reads 8 bytes at offset i as an int64. *)

end
end
module Set : sig
# 1 "environment/v1/set.mli"
(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*             Xavier Leroy, projet Cristal, INRIA Rocquencourt           *)
(*                                                                        *)
(*   Copyright 1996 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(** Sets over ordered types.

   This module implements the set data structure, given a total ordering
   function over the set elements. All operations over sets
   are purely applicative (no side-effects).
   The implementation uses balanced binary trees, and is therefore
   reasonably efficient: insertion and membership take time
   logarithmic in the size of the set, for instance.

   The [Make] functor constructs implementations for any type, given a
   [compare] function.
   For instance:
   {[
     module IntPairs =
       struct
         type t = int * int
         let compare (x0,y0) (x1,y1) =
           match Pervasives.compare x0 x1 with
               0 -> Pervasives.compare y0 y1
             | c -> c
       end

     module PairsSet = Set.Make(IntPairs)

     let m = PairsSet.(empty |> add (2,3) |> add (5,7) |> add (11,13))
   ]}

   This creates a new module [PairsSet], with a new type [PairsSet.t]
   of sets of [int * int].
*)

module type OrderedType =
  sig
    type t
      (** The type of the set elements. *)

    val compare : t -> t -> int
      (** A total ordering function over the set elements.
          This is a two-argument function [f] such that
          [f e1 e2] is zero if the elements [e1] and [e2] are equal,
          [f e1 e2] is strictly negative if [e1] is smaller than [e2],
          and [f e1 e2] is strictly positive if [e1] is greater than [e2].
          Example: a suitable ordering function is the generic structural
          comparison function {!Pervasives.compare}. *)
  end
(** Input signature of the functor {!Set.Make}. *)

module type S =
  sig
    type elt
    (** The type of the set elements. *)

    type t
    (** The type of sets. *)

    val empty: t
    (** The empty set. *)

    val is_empty: t -> bool
    (** Test whether a set is empty or not. *)

    val mem: elt -> t -> bool
    (** [mem x s] tests whether [x] belongs to the set [s]. *)

    val add: elt -> t -> t
    (** [add x s] returns a set containing all elements of [s],
       plus [x]. If [x] was already in [s], [s] is returned unchanged
       (the result of the function is then physically equal to [s]).
       @before 4.03 Physical equality was not ensured. *)

    val singleton: elt -> t
    (** [singleton x] returns the one-element set containing only [x]. *)

    val remove: elt -> t -> t
    (** [remove x s] returns a set containing all elements of [s],
       except [x]. If [x] was not in [s], [s] is returned unchanged
       (the result of the function is then physically equal to [s]).
       @before 4.03 Physical equality was not ensured. *)

    val union: t -> t -> t
    (** Set union. *)

    val inter: t -> t -> t
    (** Set intersection. *)

    val diff: t -> t -> t
    (** Set difference. *)

    val compare: t -> t -> int
    (** Total ordering between sets. Can be used as the ordering function
       for doing sets of sets. *)

    val equal: t -> t -> bool
    (** [equal s1 s2] tests whether the sets [s1] and [s2] are
       equal, that is, contain equal elements. *)

    val subset: t -> t -> bool
    (** [subset s1 s2] tests whether the set [s1] is a subset of
       the set [s2]. *)

    val iter: (elt -> unit) -> t -> unit
    (** [iter f s] applies [f] in turn to all elements of [s].
       The elements of [s] are presented to [f] in increasing order
       with respect to the ordering over the type of the elements. *)

    val map: (elt -> elt) -> t -> t
    (** [map f s] is the set whose elements are [f a0],[f a1]... [f
        aN], where [a0],[a1]...[aN] are the elements of [s].

       The elements are passed to [f] in increasing order
       with respect to the ordering over the type of the elements.

       If no element of [s] is changed by [f], [s] is returned
       unchanged. (If each output of [f] is physically equal to its
       input, the returned set is physically equal to [s].) *)

    val fold: (elt -> 'a -> 'a) -> t -> 'a -> 'a
    (** [fold f s a] computes [(f xN ... (f x2 (f x1 a))...)],
       where [x1 ... xN] are the elements of [s], in increasing order. *)

    val for_all: (elt -> bool) -> t -> bool
    (** [for_all p s] checks if all elements of the set
       satisfy the predicate [p]. *)

    val exists: (elt -> bool) -> t -> bool
    (** [exists p s] checks if at least one element of
       the set satisfies the predicate [p]. *)

    val filter: (elt -> bool) -> t -> t
    (** [filter p s] returns the set of all elements in [s]
       that satisfy predicate [p]. If [p] satisfies every element in [s],
       [s] is returned unchanged (the result of the function is then
       physically equal to [s]).
       @before 4.03 Physical equality was not ensured.*)

    val partition: (elt -> bool) -> t -> t * t
    (** [partition p s] returns a pair of sets [(s1, s2)], where
       [s1] is the set of all the elements of [s] that satisfy the
       predicate [p], and [s2] is the set of all the elements of
       [s] that do not satisfy [p]. *)

    val cardinal: t -> int
    (** Return the number of elements of a set. *)

    val elements: t -> elt list
    (** Return the list of all elements of the given set.
       The returned list is sorted in increasing order with respect
       to the ordering [Ord.compare], where [Ord] is the argument
       given to {!Set.Make}. *)

    val min_elt: t -> elt
    (** Return the smallest element of the given set
       (with respect to the [Ord.compare] ordering), or raise
       [Not_found] if the set is empty. *)

    val max_elt: t -> elt
    (** Same as {!Set.S.min_elt}, but returns the largest element of the
       given set. *)

    val choose: t -> elt
    (** Return one element of the given set, or raise [Not_found] if
       the set is empty. Which element is chosen is unspecified,
       but equal elements will be chosen for equal sets. *)

    val split: elt -> t -> t * bool * t
    (** [split x s] returns a triple [(l, present, r)], where
          [l] is the set of elements of [s] that are
          strictly less than [x];
          [r] is the set of elements of [s] that are
          strictly greater than [x];
          [present] is [false] if [s] contains no element equal to [x],
          or [true] if [s] contains an element equal to [x]. *)

    val find: elt -> t -> elt
    (** [find x s] returns the element of [s] equal to [x] (according
        to [Ord.compare]), or raise [Not_found] if no such element
        exists.
        @since 4.01.0 *)

    val of_list: elt list -> t
    (** [of_list l] creates a set from a list of elements.
        This is usually more efficient than folding [add] over the list,
        except perhaps for lists with many duplicated elements.
        @since 4.02.0 *)
  end
(** Output signature of the functor {!Set.Make}. *)

module Make (Ord : OrderedType) : S with type elt = Ord.t
(** Functor building an implementation of the set structure
   given a totally ordered type. *)
end
module Map : sig
# 1 "environment/v1/map.mli"
(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*             Xavier Leroy, projet Cristal, INRIA Rocquencourt           *)
(*                                                                        *)
(*   Copyright 1996 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(** Association tables over ordered types.

   This module implements applicative association tables, also known as
   finite maps or dictionaries, given a total ordering function
   over the keys.
   All operations over maps are purely applicative (no side-effects).
   The implementation uses balanced binary trees, and therefore searching
   and insertion take time logarithmic in the size of the map.

   For instance:
   {[
     module IntPairs =
       struct
         type t = int * int
         let compare (x0,y0) (x1,y1) =
           match Pervasives.compare x0 x1 with
               0 -> Pervasives.compare y0 y1
             | c -> c
       end

     module PairsMap = Map.Make(IntPairs)

     let m = PairsMap.(empty |> add (0,1) "hello" |> add (1,0) "world")
   ]}

   This creates a new module [PairsMap], with a new type ['a PairsMap.t]
   of maps from [int * int] to ['a]. In this example, [m] contains [string]
   values so its type is [string PairsMap.t].
*)

module type OrderedType =
  sig
    type t
      (** The type of the map keys. *)

    val compare : t -> t -> int
      (** A total ordering function over the keys.
          This is a two-argument function [f] such that
          [f e1 e2] is zero if the keys [e1] and [e2] are equal,
          [f e1 e2] is strictly negative if [e1] is smaller than [e2],
          and [f e1 e2] is strictly positive if [e1] is greater than [e2].
          Example: a suitable ordering function is the generic structural
          comparison function {!Pervasives.compare}. *)
  end
(** Input signature of the functor {!Map.Make}. *)

module type S =
  sig
    type key
    (** The type of the map keys. *)

    type (+'a) t
    (** The type of maps from type [key] to type ['a]. *)

    val empty: 'a t
    (** The empty map. *)

    val is_empty: 'a t -> bool
    (** Test whether a map is empty or not. *)

    val mem: key -> 'a t -> bool
    (** [mem x m] returns [true] if [m] contains a binding for [x],
       and [false] otherwise. *)

    val add: key -> 'a -> 'a t -> 'a t
    (** [add x y m] returns a map containing the same bindings as
       [m], plus a binding of [x] to [y]. If [x] was already bound
       in [m] to a value that is physically equal to [y],
       [m] is returned unchanged (the result of the function is
       then physically equal to [m]). Otherwise, the previous binding
       of [x] in [m] disappears.
       @before 4.03 Physical equality was not ensured. *)

    val singleton: key -> 'a -> 'a t
    (** [singleton x y] returns the one-element map that contains a binding [y]
        for [x].
        @since 3.12.0
     *)

    val remove: key -> 'a t -> 'a t
    (** [remove x m] returns a map containing the same bindings as
       [m], except for [x] which is unbound in the returned map.
       If [x] was not in [m], [m] is returned unchanged
       (the result of the function is then physically equal to [m]).
       @before 4.03 Physical equality was not ensured. *)

    val merge:
         (key -> 'a option -> 'b option -> 'c option) -> 'a t -> 'b t -> 'c t
    (** [merge f m1 m2] computes a map whose keys is a subset of keys of [m1]
        and of [m2]. The presence of each such binding, and the corresponding
        value, is determined with the function [f].
        @since 3.12.0
     *)

    val union: (key -> 'a -> 'a -> 'a option) -> 'a t -> 'a t -> 'a t
    (** [union f m1 m2] computes a map whose keys is the union of keys
        of [m1] and of [m2].  When the same binding is defined in both
        arguments, the function [f] is used to combine them.
        @since 4.03.0
    *)

    val compare: ('a -> 'a -> int) -> 'a t -> 'a t -> int
    (** Total ordering between maps.  The first argument is a total ordering
        used to compare data associated with equal keys in the two maps. *)

    val equal: ('a -> 'a -> bool) -> 'a t -> 'a t -> bool
    (** [equal cmp m1 m2] tests whether the maps [m1] and [m2] are
       equal, that is, contain equal keys and associate them with
       equal data.  [cmp] is the equality predicate used to compare
       the data associated with the keys. *)

    val iter: (key -> 'a -> unit) -> 'a t -> unit
    (** [iter f m] applies [f] to all bindings in map [m].
       [f] receives the key as first argument, and the associated value
       as second argument.  The bindings are passed to [f] in increasing
       order with respect to the ordering over the type of the keys. *)

    val fold: (key -> 'a -> 'b -> 'b) -> 'a t -> 'b -> 'b
    (** [fold f m a] computes [(f kN dN ... (f k1 d1 a)...)],
       where [k1 ... kN] are the keys of all bindings in [m]
       (in increasing order), and [d1 ... dN] are the associated data. *)

    val for_all: (key -> 'a -> bool) -> 'a t -> bool
    (** [for_all p m] checks if all the bindings of the map
        satisfy the predicate [p].
        @since 3.12.0
     *)

    val exists: (key -> 'a -> bool) -> 'a t -> bool
    (** [exists p m] checks if at least one binding of the map
        satisfy the predicate [p].
        @since 3.12.0
     *)

    val filter: (key -> 'a -> bool) -> 'a t -> 'a t
    (** [filter p m] returns the map with all the bindings in [m]
        that satisfy predicate [p]. If [p] satisfies every binding in [m],
        [m] is returned unchanged (the result of the function is then
        physically equal to [m])
        @since 3.12.0
       @before 4.03 Physical equality was not ensured.
     *)

    val partition: (key -> 'a -> bool) -> 'a t -> 'a t * 'a t
    (** [partition p m] returns a pair of maps [(m1, m2)], where
        [m1] contains all the bindings of [s] that satisfy the
        predicate [p], and [m2] is the map with all the bindings of
        [s] that do not satisfy [p].
        @since 3.12.0
     *)

    val cardinal: 'a t -> int
    (** Return the number of bindings of a map.
        @since 3.12.0
     *)

    val bindings: 'a t -> (key * 'a) list
    (** Return the list of all bindings of the given map.
       The returned list is sorted in increasing order with respect
       to the ordering [Ord.compare], where [Ord] is the argument
       given to {!Map.Make}.
        @since 3.12.0
     *)

    val min_binding: 'a t -> (key * 'a)
    (** Return the smallest binding of the given map
       (with respect to the [Ord.compare] ordering), or raise
       [Not_found] if the map is empty.
        @since 3.12.0
     *)

    val max_binding: 'a t -> (key * 'a)
    (** Same as {!Map.S.min_binding}, but returns the largest binding
        of the given map.
        @since 3.12.0
     *)

    val choose: 'a t -> (key * 'a)
    (** Return one binding of the given map, or raise [Not_found] if
       the map is empty. Which binding is chosen is unspecified,
       but equal bindings will be chosen for equal maps.
        @since 3.12.0
     *)

    val split: key -> 'a t -> 'a t * 'a option * 'a t
    (** [split x m] returns a triple [(l, data, r)], where
          [l] is the map with all the bindings of [m] whose key
        is strictly less than [x];
          [r] is the map with all the bindings of [m] whose key
        is strictly greater than [x];
          [data] is [None] if [m] contains no binding for [x],
          or [Some v] if [m] binds [v] to [x].
        @since 3.12.0
     *)

    val find: key -> 'a t -> 'a
    (** [find x m] returns the current binding of [x] in [m],
       or raises [Not_found] if no such binding exists. *)

    val map: ('a -> 'b) -> 'a t -> 'b t
    (** [map f m] returns a map with same domain as [m], where the
       associated value [a] of all bindings of [m] has been
       replaced by the result of the application of [f] to [a].
       The bindings are passed to [f] in increasing order
       with respect to the ordering over the type of the keys. *)

    val mapi: (key -> 'a -> 'b) -> 'a t -> 'b t
    (** Same as {!Map.S.map}, but the function receives as arguments both the
       key and the associated value for each binding of the map. *)


  end
(** Output signature of the functor {!Map.Make}. *)

module Make (Ord : OrderedType) : S with type key = Ord.t
(** Functor building an implementation of the map structure
   given a totally ordered type. *)
end
module Int32 : sig
# 1 "environment/v1/int32.mli"
(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*             Xavier Leroy, projet Cristal, INRIA Rocquencourt           *)
(*                                                                        *)
(*   Copyright 1996 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* TEZOS CHANGES

   * Import version 4.04.0
   * Remove deprecated functions

*)

(** 32-bit integers.

   This module provides operations on the type [int32]
   of signed 32-bit integers.  Unlike the built-in [int] type,
   the type [int32] is guaranteed to be exactly 32-bit wide on all
   platforms.  All arithmetic operations over [int32] are taken
   modulo 2{^32}.

   Performance notice: values of type [int32] occupy more memory
   space than values of type [int], and arithmetic operations on
   [int32] are generally slower than those on [int].  Use [int32]
   only when the application requires exact 32-bit arithmetic. *)

val zero : int32
(** The 32-bit integer 0. *)

val one : int32
(** The 32-bit integer 1. *)

val minus_one : int32
(** The 32-bit integer -1. *)

external neg : int32 -> int32 = "%int32_neg"
(** Unary negation. *)

external add : int32 -> int32 -> int32 = "%int32_add"
(** Addition. *)

external sub : int32 -> int32 -> int32 = "%int32_sub"
(** Subtraction. *)

external mul : int32 -> int32 -> int32 = "%int32_mul"
(** Multiplication. *)

external div : int32 -> int32 -> int32 = "%int32_div"
(** Integer division.  Raise [Division_by_zero] if the second
   argument is zero.  This division rounds the real quotient of
   its arguments towards zero, as specified for {!Pervasives.(/)}. *)

external rem : int32 -> int32 -> int32 = "%int32_mod"
(** Integer remainder.  If [y] is not zero, the result
   of [Int32.rem x y] satisfies the following property:
   [x = Int32.add (Int32.mul (Int32.div x y) y) (Int32.rem x y)].
   If [y = 0], [Int32.rem x y] raises [Division_by_zero]. *)

val succ : int32 -> int32
(** Successor.  [Int32.succ x] is [Int32.add x Int32.one]. *)

val pred : int32 -> int32
(** Predecessor.  [Int32.pred x] is [Int32.sub x Int32.one]. *)

val abs : int32 -> int32
(** Return the absolute value of its argument. *)

val max_int : int32
(** The greatest representable 32-bit integer, 2{^31} - 1. *)

val min_int : int32
(** The smallest representable 32-bit integer, -2{^31}. *)


external logand : int32 -> int32 -> int32 = "%int32_and"
(** Bitwise logical and. *)

external logor : int32 -> int32 -> int32 = "%int32_or"
(** Bitwise logical or. *)

external logxor : int32 -> int32 -> int32 = "%int32_xor"
(** Bitwise logical exclusive or. *)

val lognot : int32 -> int32
(** Bitwise logical negation *)

external shift_left : int32 -> int -> int32 = "%int32_lsl"
(** [Int32.shift_left x y] shifts [x] to the left by [y] bits.
   The result is unspecified if [y < 0] or [y >= 32]. *)

external shift_right : int32 -> int -> int32 = "%int32_asr"
(** [Int32.shift_right x y] shifts [x] to the right by [y] bits.
   This is an arithmetic shift: the sign bit of [x] is replicated
   and inserted in the vacated bits.
   The result is unspecified if [y < 0] or [y >= 32]. *)

external shift_right_logical : int32 -> int -> int32 = "%int32_lsr"
(** [Int32.shift_right_logical x y] shifts [x] to the right by [y] bits.
   This is a logical shift: zeroes are inserted in the vacated bits
   regardless of the sign of [x].
   The result is unspecified if [y < 0] or [y >= 32]. *)

external of_int : int -> int32 = "%int32_of_int"
(** Convert the given integer (type [int]) to a 32-bit integer
    (type [int32]). *)

external to_int : int32 -> int = "%int32_to_int"
(** Convert the given 32-bit integer (type [int32]) to an
   integer (type [int]).  On 32-bit platforms, the 32-bit integer
   is taken modulo 2{^31}, i.e. the high-order bit is lost
   during the conversion.  On 64-bit platforms, the conversion
   is exact. *)

external of_float : float -> int32
  = "caml_int32_of_float" "caml_int32_of_float_unboxed"
  [@@unboxed] [@@noalloc]
(** Convert the given floating-point number to a 32-bit integer,
   discarding the fractional part (truncate towards 0).
   The result of the conversion is undefined if, after truncation,
   the number is outside the range \[{!Int32.min_int}, {!Int32.max_int}\]. *)

external to_float : int32 -> float
  = "caml_int32_to_float" "caml_int32_to_float_unboxed"
  [@@unboxed] [@@noalloc]
(** Convert the given 32-bit integer to a floating-point number. *)

external of_string : string -> int32 = "caml_int32_of_string"
(** Convert the given string to a 32-bit integer.
   The string is read in decimal (by default) or in hexadecimal,
   octal or binary if the string begins with [0x], [0o] or [0b]
   respectively.
   Raise [Failure "int_of_string"] if the given string is not
   a valid representation of an integer, or if the integer represented
   exceeds the range of integers representable in type [int32]. *)

val to_string : int32 -> string
(** Return the string representation of its argument, in signed decimal. *)

external bits_of_float : float -> int32
  = "caml_int32_bits_of_float" "caml_int32_bits_of_float_unboxed"
  [@@unboxed] [@@noalloc]
(** Return the internal representation of the given float according
   to the IEEE 754 floating-point 'single format' bit layout.
   Bit 31 of the result represents the sign of the float;
   bits 30 to 23 represent the (biased) exponent; bits 22 to 0
   represent the mantissa. *)

external float_of_bits : int32 -> float
  = "caml_int32_float_of_bits" "caml_int32_float_of_bits_unboxed"
  [@@unboxed] [@@noalloc]
(** Return the floating-point number whose internal representation,
   according to the IEEE 754 floating-point 'single format' bit layout,
   is the given [int32]. *)

type t = int32
(** An alias for the type of 32-bit integers. *)

val compare: t -> t -> int
(** The comparison function for 32-bit integers, with the same specification as
    {!Pervasives.compare}.  Along with the type [t], this function [compare]
    allows the module [Int32] to be passed as argument to the functors
    {!Set.Make} and {!Map.Make}. *)

val equal: t -> t -> bool
(** The equal function for int32s.
    @since 4.03.0 *)

end
module Int64 : sig
# 1 "environment/v1/int64.mli"
(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*             Xavier Leroy, projet Cristal, INRIA Rocquencourt           *)
(*                                                                        *)
(*   Copyright 1996 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* TEZOS CHANGES

   * Import version 4.04.0
   * Remove deprecated functions

*)
(** 64-bit integers.

   This module provides operations on the type [int64] of
   signed 64-bit integers.  Unlike the built-in [int] type,
   the type [int64] is guaranteed to be exactly 64-bit wide on all
   platforms.  All arithmetic operations over [int64] are taken
   modulo 2{^64}

   Performance notice: values of type [int64] occupy more memory
   space than values of type [int], and arithmetic operations on
   [int64] are generally slower than those on [int].  Use [int64]
   only when the application requires exact 64-bit arithmetic.
*)

val zero : int64
(** The 64-bit integer 0. *)

val one : int64
(** The 64-bit integer 1. *)

val minus_one : int64
(** The 64-bit integer -1. *)

external neg : int64 -> int64 = "%int64_neg"
(** Unary negation. *)

external add : int64 -> int64 -> int64 = "%int64_add"
(** Addition. *)

external sub : int64 -> int64 -> int64 = "%int64_sub"
(** Subtraction. *)

external mul : int64 -> int64 -> int64 = "%int64_mul"
(** Multiplication. *)

external div : int64 -> int64 -> int64 = "%int64_div"
(** Integer division.  Raise [Division_by_zero] if the second
   argument is zero.  This division rounds the real quotient of
   its arguments towards zero, as specified for {!Pervasives.(/)}. *)

external rem : int64 -> int64 -> int64 = "%int64_mod"
(** Integer remainder.  If [y] is not zero, the result
   of [Int64.rem x y] satisfies the following property:
   [x = Int64.add (Int64.mul (Int64.div x y) y) (Int64.rem x y)].
   If [y = 0], [Int64.rem x y] raises [Division_by_zero]. *)

val succ : int64 -> int64
(** Successor.  [Int64.succ x] is [Int64.add x Int64.one]. *)

val pred : int64 -> int64
(** Predecessor.  [Int64.pred x] is [Int64.sub x Int64.one]. *)

val abs : int64 -> int64
(** Return the absolute value of its argument. *)

val max_int : int64
(** The greatest representable 64-bit integer, 2{^63} - 1. *)

val min_int : int64
(** The smallest representable 64-bit integer, -2{^63}. *)

external logand : int64 -> int64 -> int64 = "%int64_and"
(** Bitwise logical and. *)

external logor : int64 -> int64 -> int64 = "%int64_or"
(** Bitwise logical or. *)

external logxor : int64 -> int64 -> int64 = "%int64_xor"
(** Bitwise logical exclusive or. *)

val lognot : int64 -> int64
(** Bitwise logical negation *)

external shift_left : int64 -> int -> int64 = "%int64_lsl"
(** [Int64.shift_left x y] shifts [x] to the left by [y] bits.
   The result is unspecified if [y < 0] or [y >= 64]. *)

external shift_right : int64 -> int -> int64 = "%int64_asr"
(** [Int64.shift_right x y] shifts [x] to the right by [y] bits.
   This is an arithmetic shift: the sign bit of [x] is replicated
   and inserted in the vacated bits.
   The result is unspecified if [y < 0] or [y >= 64]. *)

external shift_right_logical : int64 -> int -> int64 = "%int64_lsr"
(** [Int64.shift_right_logical x y] shifts [x] to the right by [y] bits.
   This is a logical shift: zeroes are inserted in the vacated bits
   regardless of the sign of [x].
   The result is unspecified if [y < 0] or [y >= 64]. *)

external of_int : int -> int64 = "%int64_of_int"
(** Convert the given integer (type [int]) to a 64-bit integer
    (type [int64]). *)

external to_int : int64 -> int = "%int64_to_int"
(** Convert the given 64-bit integer (type [int64]) to an
   integer (type [int]).  On 64-bit platforms, the 64-bit integer
   is taken modulo 2{^63}, i.e. the high-order bit is lost
   during the conversion.  On 32-bit platforms, the 64-bit integer
   is taken modulo 2{^31}, i.e. the top 33 bits are lost
   during the conversion. *)

external of_float : float -> int64
  = "caml_int64_of_float" "caml_int64_of_float_unboxed"
  [@@unboxed] [@@noalloc]
(** Convert the given floating-point number to a 64-bit integer,
   discarding the fractional part (truncate towards 0).
   The result of the conversion is undefined if, after truncation,
   the number is outside the range \[{!Int64.min_int}, {!Int64.max_int}\]. *)

external to_float : int64 -> float
  = "caml_int64_to_float" "caml_int64_to_float_unboxed"
  [@@unboxed] [@@noalloc]
(** Convert the given 64-bit integer to a floating-point number. *)


external of_int32 : int32 -> int64 = "%int64_of_int32"
(** Convert the given 32-bit integer (type [int32])
   to a 64-bit integer (type [int64]). *)

external to_int32 : int64 -> int32 = "%int64_to_int32"
(** Convert the given 64-bit integer (type [int64]) to a
   32-bit integer (type [int32]). The 64-bit integer
   is taken modulo 2{^32}, i.e. the top 32 bits are lost
   during the conversion.  *)

external of_nativeint : nativeint -> int64 = "%int64_of_nativeint"
(** Convert the given native integer (type [nativeint])
   to a 64-bit integer (type [int64]). *)

external to_nativeint : int64 -> nativeint = "%int64_to_nativeint"
(** Convert the given 64-bit integer (type [int64]) to a
   native integer.  On 32-bit platforms, the 64-bit integer
   is taken modulo 2{^32}.  On 64-bit platforms,
   the conversion is exact. *)

external of_string : string -> int64 = "caml_int64_of_string"
(** Convert the given string to a 64-bit integer.
   The string is read in decimal (by default) or in hexadecimal,
   octal or binary if the string begins with [0x], [0o] or [0b]
   respectively.
   Raise [Failure "int_of_string"] if the given string is not
   a valid representation of an integer, or if the integer represented
   exceeds the range of integers representable in type [int64]. *)

val to_string : int64 -> string
(** Return the string representation of its argument, in decimal. *)

external bits_of_float : float -> int64
  = "caml_int64_bits_of_float" "caml_int64_bits_of_float_unboxed"
  [@@unboxed] [@@noalloc]
(** Return the internal representation of the given float according
   to the IEEE 754 floating-point 'double format' bit layout.
   Bit 63 of the result represents the sign of the float;
   bits 62 to 52 represent the (biased) exponent; bits 51 to 0
   represent the mantissa. *)

external float_of_bits : int64 -> float
  = "caml_int64_float_of_bits" "caml_int64_float_of_bits_unboxed"
  [@@unboxed] [@@noalloc]
(** Return the floating-point number whose internal representation,
   according to the IEEE 754 floating-point 'double format' bit layout,
   is the given [int64]. *)

type t = int64
(** An alias for the type of 64-bit integers. *)

val compare: t -> t -> int
(** The comparison function for 64-bit integers, with the same specification as
    {!Pervasives.compare}.  Along with the type [t], this function [compare]
    allows the module [Int64] to be passed as argument to the functors
    {!Set.Make} and {!Map.Make}. *)

val equal: t -> t -> bool
(** The equal function for int64s.
    @since 4.03.0 *)
end
module Buffer : sig
# 1 "environment/v1/buffer.mli"
(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*   Pierre Weis and Xavier Leroy, projet Cristal, INRIA Rocquencourt     *)
(*                                                                        *)
(*   Copyright 1999 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* TEZOS CHANGES

   * Import version 4.04.0
   * Remove channel functions

*)

(** Extensible buffers.

   This module implements buffers that automatically expand
   as necessary.  It provides accumulative concatenation of strings
   in quasi-linear time (instead of quadratic time when strings are
   concatenated pairwise).
*)

type t
(** The abstract type of buffers. *)

val create : int -> t
(** [create n] returns a fresh buffer, initially empty.
   The [n] parameter is the initial size of the internal byte sequence
   that holds the buffer contents. That byte sequence is automatically
   reallocated when more than [n] characters are stored in the buffer,
   but shrinks back to [n] characters when [reset] is called.
   For best performance, [n] should be of the same order of magnitude
   as the number of characters that are expected to be stored in
   the buffer (for instance, 80 for a buffer that holds one output
   line).  Nothing bad will happen if the buffer grows beyond that
   limit, however. In doubt, take [n = 16] for instance.
   If [n] is not between 1 and {!Sys.max_string_length}, it will
   be clipped to that interval. *)

val contents : t -> string
(** Return a copy of the current contents of the buffer.
    The buffer itself is unchanged. *)

val to_bytes : t -> bytes
(** Return a copy of the current contents of the buffer.
    The buffer itself is unchanged.
    @since 4.02 *)

val sub : t -> int -> int -> string
(** [Buffer.sub b off len] returns a copy of [len] bytes from the
    current contents of the buffer [b], starting at offset [off].

    Raise [Invalid_argument] if [srcoff] and [len] do not designate a valid
    range of [b]. *)

val blit : t -> int -> bytes -> int -> int -> unit
(** [Buffer.blit src srcoff dst dstoff len] copies [len] characters from
   the current contents of the buffer [src], starting at offset [srcoff]
   to [dst], starting at character [dstoff].

   Raise [Invalid_argument] if [srcoff] and [len] do not designate a valid
   range of [src], or if [dstoff] and [len] do not designate a valid
   range of [dst].
   @since 3.11.2
*)

val nth : t -> int -> char
(** Get the n-th character of the buffer. Raise [Invalid_argument] if
    index out of bounds *)

val length : t -> int
(** Return the number of characters currently contained in the buffer. *)

val clear : t -> unit
(** Empty the buffer. *)

val reset : t -> unit
(** Empty the buffer and deallocate the internal byte sequence holding the
   buffer contents, replacing it with the initial internal byte sequence
   of length [n] that was allocated by {!Buffer.create} [n].
   For long-lived buffers that may have grown a lot, [reset] allows
   faster reclamation of the space used by the buffer. *)

val add_char : t -> char -> unit
(** [add_char b c] appends the character [c] at the end of buffer [b]. *)

val add_string : t -> string -> unit
(** [add_string b s] appends the string [s] at the end of buffer [b]. *)

val add_bytes : t -> bytes -> unit
(** [add_bytes b s] appends the byte sequence [s] at the end of buffer [b].
    @since 4.02 *)

val add_substring : t -> string -> int -> int -> unit
(** [add_substring b s ofs len] takes [len] characters from offset
   [ofs] in string [s] and appends them at the end of buffer [b]. *)

val add_subbytes : t -> bytes -> int -> int -> unit
(** [add_subbytes b s ofs len] takes [len] characters from offset
    [ofs] in byte sequence [s] and appends them at the end of buffer [b].
    @since 4.02 *)

val add_substitute : t -> (string -> string) -> string -> unit
(** [add_substitute b f s] appends the string pattern [s] at the end
   of buffer [b] with substitution.
   The substitution process looks for variables into
   the pattern and substitutes each variable name by its value, as
   obtained by applying the mapping [f] to the variable name. Inside the
   string pattern, a variable name immediately follows a non-escaped
   [$] character and is one of the following:
   - a non empty sequence of alphanumeric or [_] characters,
   - an arbitrary sequence of characters enclosed by a pair of
   matching parentheses or curly brackets.
   An escaped [$] character is a [$] that immediately follows a backslash
   character; it then stands for a plain [$].
   Raise [Not_found] if the closing character of a parenthesized variable
   cannot be found. *)

val add_buffer : t -> t -> unit
(** [add_buffer b1 b2] appends the current contents of buffer [b2]
   at the end of buffer [b1].  [b2] is not modified. *)
end
module Format : sig
# 1 "environment/v1/format.mli"
(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*             Pierre Weis, projet Cristal, INRIA Rocquencourt            *)
(*                                                                        *)
(*   Copyright 1996 Institut National de Recherche en Informatique et     *)
(*     en Automatique.                                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

(* TEZOS CHANGES

   * Import version 4.04.0
   * Remove channel functions
   * Remove toplevel effect based functions
   * Remove deprecated functions

*)

(** Pretty printing.

   This module implements a pretty-printing facility to format values
   within 'pretty-printing boxes'. The pretty-printer splits lines
   at specified break hints, and indents lines according to the box
   structure.

   For a gentle introduction to the basics of pretty-printing using
   [Format], read
   {{:http://caml.inria.fr/resources/doc/guides/format.en.html}
    http://caml.inria.fr/resources/doc/guides/format.en.html}.

   You may consider this module as providing an extension to the
   [printf] facility to provide automatic line splitting. The addition of
   pretty-printing annotations to your regular [printf] formats gives you
   fancy indentation and line breaks.
   Pretty-printing annotations are described below in the documentation of
   the function {!Format.fprintf}.

   You may also use the explicit box management and printing functions
   provided by this module. This style is more basic but more verbose
   than the [fprintf] concise formats.

   For instance, the sequence
   [open_box 0; print_string "x ="; print_space ();
    print_int 1; close_box (); print_newline ()]
   that prints [x = 1] within a pretty-printing box, can be
   abbreviated as [printf "@[%s@ %i@]@." "x =" 1], or even shorter
   [printf "@[x =@ %i@]@." 1].

   Rule of thumb for casual users of this library:
 - use simple boxes (as obtained by [open_box 0]);
 - use simple break hints (as obtained by [print_cut ()] that outputs a
   simple break hint, or by [print_space ()] that outputs a space
   indicating a break hint);
 - once a box is opened, display its material with basic printing
   functions (e. g. [print_int] and [print_string]);
 - when the material for a box has been printed, call [close_box ()] to
   close the box;
 - at the end of your routine, flush the pretty-printer to display all the
   remaining material, e.g. evaluate [print_newline ()].

   The behaviour of pretty-printing commands is unspecified
   if there is no opened pretty-printing box. Each box opened via
   one of the [open_] functions below must be closed using [close_box]
   for proper formatting. Otherwise, some of the material printed in the
   boxes may not be output, or may be formatted incorrectly.

   In case of interactive use, the system closes all opened boxes and
   flushes all pending text (as with the [print_newline] function)
   after each phrase. Each phrase is therefore executed in the initial
   state of the pretty-printer.

   Warning: the material output by the following functions is delayed
   in the pretty-printer queue in order to compute the proper line
   splitting. Hence, you should not mix calls to the printing functions
   of the basic I/O system with calls to the functions of this module:
   this could result in some strange output seemingly unrelated with
   the evaluation order of printing commands.
*)

(** {6:tags Semantic Tags} *)

type tag = string

(** {6:meaning Changing the meaning of standard formatter pretty printing} *)

(** The [Format] module is versatile enough to let you completely redefine
  the meaning of pretty printing: you may provide your own functions to define
  how to handle indentation, line splitting, and even printing of all the
  characters that have to be printed! *)

type formatter_out_functions = {
  out_string : string -> int -> int -> unit;
  out_flush : unit -> unit;
  out_newline : unit -> unit;
  out_spaces : int -> unit;
}

(** {6:tagsmeaning Changing the meaning of printing semantic tags} *)

type formatter_tag_functions = {
  mark_open_tag : tag -> string;
  mark_close_tag : tag -> string;
  print_open_tag : tag -> unit;
  print_close_tag : tag -> unit;
}
(** The tag handling functions specific to a formatter:
  [mark] versions are the 'tag marking' functions that associate a string
  marker to a tag in order for the pretty-printing engine to flush
  those markers as 0 length tokens in the output device of the formatter.
  [print] versions are the 'tag printing' functions that can perform
  regular printing when a tag is closed or opened. *)

(** {6 Multiple formatted output} *)

type formatter
(** Abstract data corresponding to a pretty-printer (also called a
  formatter) and all its machinery.

  Defining new pretty-printers permits unrelated output of material in
  parallel on several output channels.
  All the parameters of a pretty-printer are local to a formatter:
  margin, maximum indentation limit, maximum number of boxes
  simultaneously opened, ellipsis, and so on, are specific to
  each pretty-printer and may be fixed independently.
  Given a [Pervasives.out_channel] output channel [oc], a new formatter
  writing to that channel is simply obtained by calling
  [formatter_of_out_channel oc].
  Alternatively, the [make_formatter] function allocates a new
  formatter with explicit output and flushing functions
  (convenient to output material to strings for instance).
*)

val formatter_of_buffer : Buffer.t -> formatter
(** [formatter_of_buffer b] returns a new formatter writing to
  buffer [b]. As usual, the formatter has to be flushed at
  the end of pretty printing, using [pp_print_flush] or
  [pp_print_newline], to display all the pending material. *)

val make_formatter :
  (string -> int -> int -> unit) -> (unit -> unit) -> formatter
(** [make_formatter out flush] returns a new formatter that writes according
  to the output function [out], and the flushing function [flush]. For
  instance, a formatter to the [Pervasives.out_channel] [oc] is returned by
  [make_formatter (Pervasives.output oc) (fun () -> Pervasives.flush oc)]. *)

(** {6 Basic functions to use with formatters} *)

val pp_open_hbox : formatter -> unit -> unit
val pp_open_vbox : formatter -> int -> unit
val pp_open_hvbox : formatter -> int -> unit
val pp_open_hovbox : formatter -> int -> unit
val pp_open_box : formatter -> int -> unit
val pp_close_box : formatter -> unit -> unit
val pp_open_tag : formatter -> string -> unit
val pp_close_tag : formatter -> unit -> unit
val pp_print_string : formatter -> string -> unit
val pp_print_as : formatter -> int -> string -> unit
val pp_print_int : formatter -> int -> unit
val pp_print_float : formatter -> float -> unit
val pp_print_char : formatter -> char -> unit
val pp_print_bool : formatter -> bool -> unit
val pp_print_break : formatter -> int -> int -> unit
val pp_print_cut : formatter -> unit -> unit
val pp_print_space : formatter -> unit -> unit
val pp_force_newline : formatter -> unit -> unit
val pp_print_flush : formatter -> unit -> unit
val pp_print_newline : formatter -> unit -> unit
val pp_print_if_newline : formatter -> unit -> unit
val pp_set_tags : formatter -> bool -> unit
val pp_set_print_tags : formatter -> bool -> unit
val pp_set_mark_tags : formatter -> bool -> unit
val pp_get_print_tags : formatter -> unit -> bool
val pp_get_mark_tags : formatter -> unit -> bool
val pp_set_margin : formatter -> int -> unit
val pp_get_margin : formatter -> unit -> int
val pp_set_max_indent : formatter -> int -> unit
val pp_get_max_indent : formatter -> unit -> int
val pp_set_max_boxes : formatter -> int -> unit
val pp_get_max_boxes : formatter -> unit -> int
val pp_over_max_boxes : formatter -> unit -> bool
val pp_set_ellipsis_text : formatter -> string -> unit
val pp_get_ellipsis_text : formatter -> unit -> string

val pp_set_formatter_output_functions :
  formatter -> (string -> int -> int -> unit) -> (unit -> unit) -> unit

val pp_get_formatter_output_functions :
  formatter -> unit -> (string -> int -> int -> unit) * (unit -> unit)

val pp_set_formatter_tag_functions :
  formatter -> formatter_tag_functions -> unit

val pp_get_formatter_tag_functions :
  formatter -> unit -> formatter_tag_functions

val pp_set_formatter_out_functions :
  formatter -> formatter_out_functions -> unit

val pp_get_formatter_out_functions :
  formatter -> unit -> formatter_out_functions
(** These functions are the basic ones: usual functions
   operating on the standard formatter are defined via partial
   evaluation of these primitives. For instance,
   [print_string] is equal to [pp_print_string std_formatter]. *)

val pp_flush_formatter : formatter -> unit
(** [pp_flush_formatter fmt] flushes [fmt]'s internal queue, ensuring that all
    the printing and flushing actions have been performed. In addition, this
    operation will close all boxes and reset the state of the formatter.

    This will not flush [fmt]'s output. In most cases, the user may want to use
    {!pp_print_flush} instead. *)

(** {6 Convenience formatting functions.} *)

val pp_print_list:
  ?pp_sep:(formatter -> unit -> unit) ->
  (formatter -> 'a -> unit) -> (formatter -> 'a list -> unit)
(** [pp_print_list ?pp_sep pp_v ppf l] prints items of list [l],
  using [pp_v] to print each item, and calling [pp_sep]
  between items ([pp_sep] defaults to {!pp_print_cut}).
  Does nothing on empty lists.

  @since 4.02.0
*)

val pp_print_text : formatter -> string -> unit
(** [pp_print_text ppf s] prints [s] with spaces and newlines
  respectively printed with {!pp_print_space} and
  {!pp_force_newline}.

  @since 4.02.0
*)

(** {6 [printf] like functions for pretty-printing.} *)

val fprintf : formatter -> ('a, formatter, unit) format -> 'a

(** [fprintf ff fmt arg1 ... argN] formats the arguments [arg1] to [argN]
  according to the format string [fmt], and outputs the resulting string on
  the formatter [ff].

  The format [fmt] is a character string which contains three types of
  objects: plain characters and conversion specifications as specified in
  the [Printf] module, and pretty-printing indications specific to the
  [Format] module.

  The pretty-printing indication characters are introduced by
  a [@] character, and their meanings are:
  - [@\[]: open a pretty-printing box. The type and offset of the
    box may be optionally specified with the following syntax:
    the [<] character, followed by an optional box type indication,
    then an optional integer offset, and the closing [>] character.
    Box type is one of [h], [v], [hv], [b], or [hov].
    '[h]' stands for an 'horizontal' box,
    '[v]' stands for a 'vertical' box,
    '[hv]' stands for an 'horizontal-vertical' box,
    '[b]' stands for an 'horizontal-or-vertical' box demonstrating indentation,
    '[hov]' stands a simple 'horizontal-or-vertical' box.
    For instance, [@\[<hov 2>] opens an 'horizontal-or-vertical'
    box with indentation 2 as obtained with [open_hovbox 2].
    For more details about boxes, see the various box opening
    functions [open_*box].
  - [@\]]: close the most recently opened pretty-printing box.
  - [@,]: output a 'cut' break hint, as with [print_cut ()].
  - [@ ]: output a 'space' break hint, as with [print_space ()].
  - [@;]: output a 'full' break hint as with [print_break]. The
    [nspaces] and [offset] parameters of the break hint may be
    optionally specified with the following syntax:
    the [<] character, followed by an integer [nspaces] value,
    then an integer [offset], and a closing [>] character.
    If no parameters are provided, the good break defaults to a
    'space' break hint.
  - [@.]: flush the pretty printer and split the line, as with
    [print_newline ()].
  - [@<n>]: print the following item as if it were of length [n].
    Hence, [printf "@<0>%s" arg] prints [arg] as a zero length string.
    If [@<n>] is not followed by a conversion specification,
    then the following character of the format is printed as if
    it were of length [n].
  - [@\{]: open a tag. The name of the tag may be optionally
    specified with the following syntax:
    the [<] character, followed by an optional string
    specification, and the closing [>] character. The string
    specification is any character string that does not contain the
    closing character ['>']. If omitted, the tag name defaults to the
    empty string.
    For more details about tags, see the functions [open_tag] and
    [close_tag].
  - [@\}]: close the most recently opened tag.
  - [@?]: flush the pretty printer as with [print_flush ()].
    This is equivalent to the conversion [%!].
  - [@\n]: force a newline, as with [force_newline ()], not the normal way
    of pretty-printing, you should prefer using break hints inside a vertical
    box.

  Note: If you need to prevent the interpretation of a [@] character as a
  pretty-printing indication, you must escape it with a [%] character.
  Old quotation mode [@@] is deprecated since it is not compatible with
  formatted input interpretation of character ['@'].

  Example: [printf "@[%s@ %d@]@." "x =" 1] is equivalent to
  [open_box (); print_string "x ="; print_space ();
   print_int 1; close_box (); print_newline ()].
  It prints [x = 1] within a pretty-printing 'horizontal-or-vertical' box.

*)

val sprintf : ('a, unit, string) format -> 'a
(** Same as [printf] above, but instead of printing on a formatter,
  returns a string containing the result of formatting the arguments.
  Note that the pretty-printer queue is flushed at the end of {e each
  call} to [sprintf].

  In case of multiple and related calls to [sprintf] to output
  material on a single string, you should consider using [fprintf]
  with the predefined formatter [str_formatter] and call
  [flush_str_formatter ()] to get the final result.

  Alternatively, you can use [Format.fprintf] with a formatter writing to a
  buffer of your own: flushing the formatter and the buffer at the end of
  pretty-printing returns the desired string.
*)

val asprintf : ('a, formatter, unit, string) format4 -> 'a
(** Same as [printf] above, but instead of printing on a formatter,
  returns a string containing the result of formatting the arguments.
  The type of [asprintf] is general enough to interact nicely with [%a]
  conversions.
  @since 4.01.0
*)

val ifprintf : formatter -> ('a, formatter, unit) format -> 'a
(** Same as [fprintf] above, but does not print anything.
  Useful to ignore some material when conditionally printing.
  @since 3.10.0
*)

(** Formatted output functions with continuations. *)

val kfprintf :
  (formatter -> 'a) -> formatter ->
  ('b, formatter, unit, 'a) format4 -> 'b
(** Same as [fprintf] above, but instead of returning immediately,
  passes the formatter to its first argument at the end of printing. *)

val ikfprintf :
  (formatter -> 'a) -> formatter ->
  ('b, formatter, unit, 'a) format4 -> 'b
(** Same as [kfprintf] above, but does not print anything.
  Useful to ignore some material when conditionally printing.
  @since 3.12.0
*)

val ksprintf : (string -> 'a) -> ('b, unit, string, 'a) format4 -> 'b
(** Same as [sprintf] above, but instead of returning the string,
  passes it to the first argument. *)

val kasprintf : (string -> 'a) -> ('b, formatter, unit, 'a) format4 -> 'b
(** Same as [asprintf] above, but instead of returning the string,
  passes it to the first argument.
  @since 4.03
*)
end
module Z : sig
# 1 "environment/v1/z.mli"
(**************************************************************************)
(*                                                                        *)
(*    Copyright (c) 2014 - 2016.                                          *)
(*    Dynamic Ledger Solutions, Inc. <contact@tezos.com>                  *)
(*                                                                        *)
(*    All rights reserved. No warranty, explicit or implicit, provided.   *)
(*                                                                        *)
(**************************************************************************)

type t
val zero: t
val one: t

external abs: t -> t = "ml_z_abs" "ml_as_z_abs"
(** Absolute value. *)

external neg: t -> t = "ml_z_neg" "ml_as_z_neg"
(** Unary negation. *)

external add: t -> t -> t = "ml_z_add" "ml_as_z_add"
(** Addition. *)

external sub: t -> t -> t = "ml_z_sub" "ml_as_z_sub"
(** Subtraction. *)

external mul: t -> t -> t = "ml_z_mul" "ml_as_z_mul"
(** Multiplication. *)

val ediv_rem: t -> t -> (t * t)
(** Euclidean division and remainder.  [ediv_rem a b] returns a pair [(q, r)]
    such that [a = b * q + r] and [0 <= r < |b|].
    Raises [Division_by_zero] if [b = 0].
 *)

external logand: t -> t -> t = "ml_z_logand" "ml_as_z_logand"
(** Bitwise logical and. *)

external logor: t -> t -> t = "ml_z_logor" "ml_as_z_logor"
(** Bitwise logical or. *)

external logxor: t -> t -> t = "ml_z_logxor" "ml_as_z_logxor"
(** Bitwise logical exclusive or. *)

external lognot: t -> t = "ml_z_lognot" "ml_as_z_lognot"
(** Bitwise logical negation.
    The identity [lognot a]=[-a-1] always hold.
 *)

external shift_left: t -> int -> t = "ml_z_shift_left" "ml_as_z_shift_left"
(** Shifts to the left.
    Equivalent to a multiplication by a power of 2.
    The second argument must be non-negative.
 *)

external shift_right: t -> int -> t = "ml_z_shift_right" "ml_as_z_shift_right"
(** Shifts to the right.
    This is an arithmetic shift,
    equivalent to a division by a power of 2 with rounding towards -oo.
    The second argument must be non-negative.
 *)

val to_string: t -> string
val of_string: string -> t

external to_int64: t -> int64 = "ml_z_to_int64"
(** Converts to a 64-bit integer. May raise [Overflow]. *)
external of_int64: int64 -> t = "ml_z_of_int64"
(** Converts from a 64-bit integer. *)

external to_int: t -> int = "ml_z_to_int"
(** Converts to a base integer. May raise an [Overflow]. *)
external of_int: int -> t = "ml_z_of_int" [@@ noalloc]
(** Converts from a base integer. *)

external equal: t -> t -> bool = "ml_z_equal" [@@ noalloc]
external compare: t -> t -> int = "ml_z_compare" [@@ noalloc]
end
module Lwt_sequence : sig
# 1 "environment/v1/lwt_sequence.mli"
(* Lightweight thread library for OCaml
 * http://www.ocsigen.org/lwt
 * Interface Lwt_sequence
 * Copyright (C) 2009 Jérémie Dimino
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, with linking exceptions;
 * either version 2.1 of the License, or (at your option) any later
 * version. See COPYING file for details.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
 * 02111-1307, USA.
 *)

(** Mutable sequence of elements *)

(** A sequence is an object holding a list of elements which support
    the following operations:

    - adding an element to the left or the right in time and space O(1)
    - taking an element from the left or the right in time and space O(1)
    - removing a previously added element from a sequence in time and space O(1)
    - removing an element while the sequence is being transversed.
*)

type 'a t
  (** Type of a sequence holding values of type ['a] *)

type 'a node
  (** Type of a node holding one value of type ['a] in a sequence *)

(** {2 Operation on nodes} *)

val get : 'a node -> 'a
  (** Returns the contents of a node *)

val set : 'a node -> 'a -> unit
  (** Change the contents of a node *)

val remove : 'a node -> unit
  (** Removes a node from the sequence it is part of. It does nothing
      if the node has already been removed. *)

(** {2 Operations on sequence} *)

val create : unit -> 'a t
  (** [create ()] creates a new empty sequence *)

val is_empty : 'a t -> bool
  (** Returns [true] iff the given sequence is empty *)

val length : 'a t -> int
  (** Returns the number of elemenets in the given sequence. This is a
      O(n) operation where [n] is the number of elements in the
      sequence. *)

val add_l : 'a -> 'a t -> 'a node
  (** [add_l x s] adds [x] to the left of the sequence [s] *)

val add_r : 'a -> 'a t -> 'a node
  (** [add_l x s] adds [x] to the right of the sequence [s] *)

exception Empty
  (** Exception raised by [take_l] and [tale_s] and when the sequence
      is empty *)

val take_l : 'a t -> 'a
  (** [take_l x s] remove and returns the leftmost element of [s]

      @raise Empty if the sequence is empty *)

val take_r : 'a t -> 'a
  (** [take_l x s] remove and returns the rightmost element of [s]

      @raise Empty if the sequence is empty *)

val take_opt_l : 'a t -> 'a option
  (** [take_opt_l x s] remove and returns [Some x] where [x] is the
      leftmost element of [s] or [None] if [s] is empty *)

val take_opt_r : 'a t -> 'a option
  (** [take_opt_l x s] remove and returns [Some x] where [x] is the
      rightmost element of [s] or [None] if [s] is empty *)

val transfer_l : 'a t -> 'a t -> unit
  (** [transfer_l s1 s2] removes all elements of [s1] and add them at
      the left of [s2]. This operation runs in constant time and
      space. *)

val transfer_r : 'a t -> 'a t -> unit
  (** [transfer_r s1 s2] removes all elements of [s1] and add them at
      the right of [s2]. This operation runs in constant time and
      space. *)

(** {2 Sequence iterators} *)

(** Note: it is OK to remove a node while traversing a sequence *)

val iter_l : ('a -> unit) -> 'a t -> unit
  (** [iter_l f s] applies [f] on all elements of [s] starting from
      the left *)

val iter_r : ('a -> unit) -> 'a t -> unit
  (** [iter_l f s] applies [f] on all elements of [s] starting from
      the right *)

val iter_node_l : ('a node -> unit) -> 'a t -> unit
  (** [iter_l f s] applies [f] on all nodes of [s] starting from
      the left *)

val iter_node_r : ('a node -> unit) -> 'a t -> unit
  (** [iter_l f s] applies [f] on all nodes of [s] starting from
      the right *)

val fold_l : ('a -> 'b -> 'b) -> 'a t -> 'b -> 'b
  (** [fold_l f s] is:
      {[
        fold_l f s x = f en (... (f e2 (f e1 x)))
      ]}
      where [e1], [e2], ..., [en] are the elements of [s]
  *)

val fold_r : ('a -> 'b -> 'b) -> 'a t -> 'b -> 'b
  (** [fold_r f s] is:
      {[
        fold_r f s x = f e1 (f e2 (... (f en x)))
      ]}
      where [e1], [e2], ..., [en] are the elements of [s]
  *)

val find_node_opt_l : ('a -> bool) -> 'a t -> 'a node option
  (** [find_node_opt_l f s] returns [Some x], where [x] is the first node of
      [s] starting from the left that satisfies [f] or [None] if none
      exists. *)

val find_node_opt_r : ('a -> bool) -> 'a t -> 'a node option
  (** [find_node_opt_r f s] returns [Some x], where [x] is the first node of
      [s] starting from the right that satisfies [f] or [None] if none
      exists. *)

val find_node_l : ('a -> bool) -> 'a t -> 'a node
  (** [find_node_l f s] returns the first node of [s] starting from the left
      that satisfies [f] or raises [Not_found] if none exists. *)

val find_node_r : ('a -> bool) -> 'a t -> 'a node
  (** [find_node_r f s] returns the first node of [s] starting from the right
      that satisfies [f] or raises [Not_found] if none exists. *)
end
module Lwt : sig
# 1 "environment/v1/lwt.mli"
(* Lightweight thread library for OCaml
 * http://www.ocsigen.org/lwt
 * Interface Lwt
 * Copyright (C) 2005-2008 J�r�me Vouillon
 * Laboratoire PPS - CNRS Universit� Paris Diderot
 *               2009-2012 J�r�mie Dimino
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, with linking exceptions;
 * either version 2.1 of the License, or (at your option) any later
 * version. See COPYING file for details.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
 * 02111-1307, USA.
 *)

(* TEZOS CHANGES

   * import version 2.4.5
   * Comment a few function that shouldn't be used in the protocol:
     * choose: scheduling may be system dependent.
     * wait/wakeup
     * state
     * cancel
     * pause
     * async
     * thread storage
     * lwt exceptions
*)


(** Module [Lwt]: cooperative light-weight threads. *)

(** This module defines {e cooperative light-weight threads} with
    their primitives. A {e light-weight thread} represent a
    computation that may be not terminated, for example because it is
    waiting for some event to happen.

    Lwt threads are cooperative in the sense that switching to another
    thread is awlays explicit (with {!wakeup} or {!wakeup_exn}). When a
    thread is running, it executes as much as possible, and then
    returns (a value or an eror) or sleeps.

    Note that inside a Lwt thread, exceptions must be raised with
    {!fail} instead of [raise]. Also the [try ... with ...]
    construction will not catch Lwt errors. You must use {!catch}
    instead. You can also use {!wrap} for functions that may raise
    normal exception.

    Lwt also provides the syntax extension {!Pa_lwt} to make code
    using Lwt more readable.
*)

(** {2 Definitions and basics} *)

type +'a t
  (** The type of threads returning a result of type ['a]. *)

val return : 'a -> 'a t
  (** [return e] is a thread whose return value is the value of the
      expression [e]. *)

(* val fail : exn -> 'a t *)
(*   (\** [fail e] is a thread that fails with the exception [e]. *\) *)

val bind : 'a t -> ('a -> 'b t) -> 'b t
  (** [bind t f] is a thread which first waits for the thread [t] to
      terminate and then, if the thread succeeds, behaves as the
      application of function [f] to the return value of [t].  If the
      thread [t] fails, [bind t f] also fails, with the same
      exception.

      The expression [bind t (fun x -> t')] can intuitively be read as
      [let x = t in t'], and if you use the {e lwt.syntax} syntax
      extension, you can write a bind operation like that: [lwt x = t in t'].

      Note that [bind] is also often used just for synchronization
      purpose: [t'] will not execute before [t] is terminated.

      The result of a thread can be bound several time. *)

val (>>=) : 'a t -> ('a -> 'b t) -> 'b t
  (** [t >>= f] is an alternative notation for [bind t f]. *)

val (=<<) : ('a -> 'b t) -> 'a t -> 'b t
  (** [f =<< t] is [t >>= f] *)

val map : ('a -> 'b) -> 'a t -> 'b t
  (** [map f m] map the result of a thread. This is the same as [bind
      m (fun x -> return (f x))] *)

val (>|=) : 'a t -> ('a -> 'b) -> 'b t
  (** [m >|= f] is [map f m] *)

val (=|<) : ('a -> 'b) -> 'a t -> 'b t
  (** [f =|< m] is [map f m] *)

(** {3 Pre-allocated threads} *)

val return_unit : unit t
  (** [return_unit = return ()] *)

val return_none : 'a option t
  (** [return_none = return None] *)

val return_nil : 'a list t
  (** [return_nil = return \[\]] *)

val return_true : bool t
  (** [return_true = return true] *)

val return_false : bool t
  (** [return_false = return false] *)

(* (\** {2 Thread storage} *\) *)

(* type 'a key *)
(*   (\** Type of a key. Keys are used to store local values into *)
(*       threads *\) *)

(* val new_key : unit -> 'a key *)
(*   (\** [new_key ()] creates a new key. *\) *)

(* val get : 'a key -> 'a option *)
(*   (\** [get key] returns the value associated with [key] in the current *)
(*       thread. *\) *)

(* val with_value : 'a key -> 'a option -> (unit -> 'b) -> 'b *)
(*   (\** [with_value key value f] executes [f] with [value] associated to *)
(*       [key]. The previous value associated to [key] is restored after *)
(*       [f] terminates. *\) *)

(* (\** {2 Exceptions handling} *\) *)

(* val catch : (unit -> 'a t) -> (exn -> 'a t) -> 'a t *)
(*   (\** [catch t f] is a thread that behaves as the thread [t ()] if *)
(*       this thread succeeds.  If the thread [t ()] fails with some *)
(*       exception, [catch t f] behaves as the application of [f] to this *)
(*       exception. *\) *)

(* val try_bind : (unit -> 'a t) -> ('a -> 'b t) -> (exn -> 'b t) -> 'b t *)
(*   (\** [try_bind t f g] behaves as [bind (t ()) f] if [t] does not *)
(*       fail.  Otherwise, it behaves as the application of [g] to the *)
(*       exception associated to [t ()]. *\) *)

(* val finalize : (unit -> 'a t) -> (unit -> unit t) -> 'a t *)
(*   (\** [finalize f g] returns the same result as [f ()] whether it *)
(*       fails or not. In both cases, [g ()] is executed after [f]. *\) *)

(* val wrap : (unit -> 'a) -> 'a t *)
(*   (\** [wrap f] calls [f] and transform the result into a monad. If [f] *)
(*       raise an exception, it is catched by Lwt. *)

(*       This is actually the same as: *)

(*       {[ *)
(*         try *)
(*           return (f ()) *)
(*         with exn -> *)
(*           fail exn *)
(*       ]} *)
(*   *\) *)

(* val wrap1 : ('a -> 'b) -> 'a -> 'b t *)
(*   (\** [wrap1 f x] applies [f] on [x] and returns the result as a *)
(*       thread. If the application of [f] to [x] raise an exception it *)
(*       is catched and a thread is returned. *)

(*       Note that you must use {!wrap} instead of {!wrap1} if the *)
(*       evaluation of [x] may raise an exception. *)

(*       for example the following code is not ok: *)

(*       {[ *)
(*         wrap1 f (Hashtbl.find table key) *)
(*       ]} *)

(*       you should write instead: *)

(*       {[ *)
(*         wrap (fun () -> f (Hashtbl.find table key)) *)
(*       ]} *)
(*   *\) *)

(* val wrap2 : ('a -> 'b -> 'c) -> 'a -> 'b -> 'c t *)
(* val wrap3 : ('a -> 'b -> 'c -> 'd) -> 'a -> 'b -> 'c -> 'd t *)
(* val wrap4 : ('a -> 'b -> 'c -> 'd -> 'e) -> 'a -> 'b -> 'c -> 'd -> 'e t *)
(* val wrap5 : ('a -> 'b -> 'c -> 'd -> 'e -> 'f) -> 'a -> 'b -> 'c -> 'd -> 'e -> 'f t *)
(* val wrap6 : ('a -> 'b -> 'c -> 'd -> 'e -> 'f -> 'g) -> 'a -> 'b -> 'c -> 'd -> 'e -> 'f -> 'g t *)
(* val wrap7 : ('a -> 'b -> 'c -> 'd -> 'e -> 'f -> 'g -> 'h) -> 'a -> 'b -> 'c -> 'd -> 'e -> 'f -> 'g -> 'h t *)

(** {2 Multi-threads composition} *)

(* we shouldn't use choose: the scheduling may be system dependent *)

(* val choose : 'a t list -> 'a t *)
(*   (\** [choose l] behaves as the first thread in [l] to terminate.  If *)
(*       several threads are already terminated, one is choosen at *)
(*       random. *)

(*       Note: {!choose} leaves the local values of the current thread *)
(*       unchanged. *\) *)

(* val nchoose : 'a t list -> 'a list t *)
(*   (\** [nchoose l] returns the value of all that have succcessfully *)
(*       terminated. If all threads are sleeping, it waits for at least *)
(*       one to terminates. If one the threads of [l] fails, [nchoose] *)
(*       fails with the same exception. *)

(*       Note: {!nchoose} leaves the local values of the current thread *)
(*       unchanged. *\) *)

(* val nchoose_split : 'a t list -> ('a list * 'a t list) t *)
(*   (\** [nchoose_split l] does the same as {!nchoose} but also retrurns *)
(*       the list of threads that have not yet terminated. *\) *)

val join : unit t list -> unit t
  (** [join l] waits for all threads in [l] to terminate. If one of
      the threads fails, then [join l] will fails with the same
      exception as the first one to terminate.

      Note: {!join} leaves the local values of the current thread
      unchanged. *)

(* val ( <?> ) : 'a t -> 'a t -> 'a t *)
(*   (\** [t <?> t'] is the same as [choose [t; t']] *\) *)

val ( <&> ) : unit t -> unit t -> unit t
  (** [t <&> t'] is the same as [join [t; t']] *)

(* val async : (unit -> 'a t) -> unit *)
(*   (\** [async f] starts a thread without waiting for the result. If it *)
(*       fails (now or later), the exception is given to *)
(*       {!async_exception_hook}. *)

(*       You should use this function if you want to start a thread that *)
(*       might fail and don't care what its return value is, nor when it *)
(*       terminates (for instance, because it is looping). *\) *)

(* val ignore_result : 'a t -> unit *)
(*   (\** [ignore_result t] is like [Pervasives.ignore t] except that: *)

(*       - if [t] already failed, it raises the exception now, *)
(*       - if [t] is sleeping and fails later, the exception will be *)
(*         given to {!async_exception_hook}. *\) *)

(* val async_exception_hook : (exn -> unit) ref *)
(*   (\** Function called when a asynchronous exception is thrown. *)

(*       The default behavior is to print an error message with a *)
(*       backtrace if available and to exit the program. *)

(*       The behavior is undefined if this function raise an *)
(*       exception. *\) *)

(* (\** {2 Sleeping and resuming} *\) *)

(* type 'a u *)
(*   (\** The type of thread wakeners. *\) *)

(* val wait : unit -> 'a t * 'a u *)
(*   (\** [wait ()] is a pair of a thread which sleeps forever (unless it *)
(*       is resumed by one of the functions [wakeup], [wakeup_exn] below) *)
(*       and the corresponding wakener.  This thread does not block the *)
(*       execution of the remainder of the program (except of course, if *)
(*       another thread tries to wait for its termination). *\) *)

(* val wakeup : 'a u -> 'a -> unit *)
(*   (\** [wakeup t e] makes the sleeping thread [t] terminate and return *)
(*       the value of the expression [e]. *\) *)

(* val wakeup_exn : 'a u -> exn -> unit *)
(*   (\** [wakeup_exn t e] makes the sleeping thread [t] fail with the *)
(*       exception [e]. *\) *)

(* val wakeup_later : 'a u -> 'a -> unit *)
(*   (\** Same as {!wakeup} but it is not guaranteed that the thread will *)
(*       be woken up immediately. *\) *)

(* val wakeup_later_exn : 'a u -> exn -> unit *)
(*   (\** Same as {!wakeup_exn} but it is not guaranteed that the thread *)
(*       will be woken up immediately. *\) *)

(* val waiter_of_wakener : 'a u -> 'a t *)
(*   (\** Returns the thread associated to a wakener. *\) *)

(* type +'a result *)
(*   (\** Either a value of type ['a], either an exception. *\) *)

(* val make_value : 'a -> 'a result *)
(*   (\** [value x] creates a result containing the value [x]. *\) *)

(* val make_error : exn -> 'a result *)
(*   (\** [error e] creates a result containing the exception [e]. *\) *)

(* val of_result : 'a result -> 'a t *)
(*   (\** Returns a thread from a result. *\) *)

(* val wakeup_result : 'a u -> 'a result -> unit *)
(*   (\** [wakeup_result t r] makes the sleeping thread [t] terminate with *)
(*       the result [r]. *\) *)

(* val wakeup_later_result : 'a u -> 'a result -> unit *)
(*   (\** Same as {!wakeup_result} but it is not guaranteed that the *)
(*       thread will be woken up immediately. *\) *)

(* (\** {2 Threads state} *\) *)

(* (\** State of a thread *\) *)
(* type 'a state = *)
(*   | Return of 'a *)
(*       (\** The thread which has successfully terminated *\) *)
(*   | Fail of exn *)
(*       (\** The thread raised an exception *\) *)
(*   | Sleep *)
(*       (\** The thread is sleeping *\) *)

(* val state : 'a t -> 'a state *)
(*   (\** [state t] returns the state of a thread *\) *)

(* val is_sleeping : 'a t -> bool *)
(*   (\** [is_sleeping t] returns [true] iff [t] is sleeping. *\) *)

(* (\** {2 Cancelable threads} *\) *)

(* (\** Cancelable threads are the same as regular threads except that *)
(*     they can be canceled. *\) *)

(* exception Canceled *)
(*   (\** Canceled threads fails with this exception *\) *)

(* val task : unit -> 'a t * 'a u *)
(*   (\** [task ()] is the same as [wait ()] except that threads created *)
(*       with [task] can be canceled. *\) *)

(* val on_cancel : 'a t -> (unit -> unit) -> unit *)
(*   (\** [on_cancel t f] executes [f] when [t] is canceled. [f] will be *)
(*       executed before all other threads waiting on [t]. *)

(*       If [f] raises an exception it is given to *)
(*       {!async_exception_hook}. *\) *)

(* val add_task_r : 'a u Lwt_sequence.t -> 'a t *)
(*   (\** [add_task_r seq] creates a sleeping thread, adds its wakener to *)
(*       the right of [seq] and returns its waiter. When the thread is *)
(*       canceled, it is removed from [seq]. *\) *)

(* val add_task_l : 'a u Lwt_sequence.t -> 'a t *)
(*   (\** [add_task_l seq] creates a sleeping thread, adds its wakener to *)
(*       the left of [seq] and returns its waiter. When the thread is *)
(*       canceled, it is removed from [seq]. *\) *)

(* val cancel : 'a t -> unit *)
(*   (\** [cancel t] cancels the threads [t]. This means that the deepest *)
(*       sleeping thread created with [task] and connected to [t] is *)
(*       woken up with the exception {!Canceled}. *)

(*       For example, in the following code: *)

(*       {[ *)
(*         let waiter, wakener = task () in *)
(*         cancel (waiter >> printl "plop") *)
(*       ]} *)

(*       [waiter] will be woken up with {!Canceled}. *)
(*   *\) *)

(* val pick : 'a t list -> 'a t *)
(*   (\** [pick l] is the same as {!choose}, except that it cancels all *)
(*       sleeping threads when one terminates. *)

(*       Note: {!pick} leaves the local values of the current thread *)
(*       unchanged. *\) *)

(* val npick : 'a t list -> 'a list t *)
(*   (\** [npick l] is the same as {!nchoose}, except that it cancels all *)
(*       sleeping threads when one terminates. *)

(*       Note: {!npick} leaves the local values of the current thread *)
(*       unchanged. *\) *)

(* val protected : 'a t -> 'a t *)
(*   (\** [protected thread] creates a new cancelable thread which behave *)
(*       as [thread] except that cancelling it does not cancel *)
(*       [thread]. *\) *)

(* val no_cancel : 'a t -> 'a t *)
(*   (\** [no_cancel thread] creates a thread which behave as [thread] *)
(*       except that it cannot be canceled. *\) *)

(* (\** {2 Pause} *\) *)

(* val pause : unit -> unit t *)
(*   (\** [pause ()] is a sleeping thread which is wake up on the next *)
(*       call to {!wakeup_paused}. A thread created with [pause] can be *)
(*       canceled. *\) *)

(* val wakeup_paused : unit -> unit *)
(*   (\** [wakeup_paused ()] wakes up all threads which suspended *)
(*       themselves with {!pause}. *)

(*       This function is called by the scheduler, before entering the *)
(*       main loop. You usually do not have to call it directly, except *)
(*       if you are writing a custom scheduler. *)

(*       Note that if a paused thread resumes and pauses again, it will not *)
(*       be woken up at this point. *\) *)

(* val paused_count : unit -> int *)
(*   (\** [paused_count ()] returns the number of currently paused *)
(*       threads. *\) *)

(* val register_pause_notifier : (int -> unit) -> unit *)
(*   (\** [register_pause_notifier f] register a function [f] that will be *)
(*       called each time pause is called. The parameter passed to [f] is *)
(*       the new number of threads paused. It is usefull to be able to *)
(*       call {!wakeup_paused} when there is no scheduler *\) *)

(* (\** {2 Misc} *\) *)

(* val on_success : 'a t -> ('a -> unit) -> unit *)
(*   (\** [on_success t f] executes [f] when [t] terminates without *)
(*       failing. If [f] raises an exception it is given to *)
(*       {!async_exception_hook}. *\) *)

(* val on_failure : 'a t -> (exn -> unit) -> unit *)
(*   (\** [on_failure t f] executes [f] when [t] terminates and fails. If *)
(*       [f] raises an exception it is given to *)
(*       {!async_exception_hook}. *\) *)

(* val on_termination : 'a t -> (unit -> unit) -> unit *)
(*   (\** [on_termination t f] executes [f] when [t] terminates. If [f] *)
(*       raises an exception it is given to {!async_exception_hook}. *\) *)

(* val on_any : 'a t -> ('a -> unit) -> (exn -> unit) -> unit *)
(*   (\** [on_any t f g] executes [f] or [g] when [t] terminates. If [f] *)
(*       or [g] raises an exception it is given to *)
(*       {!async_exception_hook}. *\) *)

(* (\**/**\) *)

(* (\* The functions below are probably not useful for the casual user. *)
(*    They provide the basic primitives on which can be built multi- *)
(*    threaded libraries such as Lwt_unix. *\) *)

(* val poll : 'a t -> 'a option *)
(*       (\* [poll e] returns [Some v] if the thread [e] is terminated and *)
(*          returned the value [v].  If the thread failed with some *)
(*          exception, this exception is raised.  If the thread is still *)
(*          running, [poll e] returns [None] without blocking. *\) *)

(* val apply : ('a -> 'b t) -> 'a -> 'b t *)
(*       (\* [apply f e] apply the function [f] to the expression [e].  If *)
(*          an exception is raised during this application, it is caught *)
(*          and the resulting thread fails with this exception. *\) *)
(* (\* Q: Could be called 'glue' or 'trap' or something? *\) *)

(* val backtrace_bind : (exn -> exn) -> 'a t -> ('a -> 'b t) -> 'b t *)
(* val backtrace_catch : (exn -> exn) -> (unit -> 'a t) -> (exn -> 'a t) -> 'a t *)
(* val backtrace_try_bind : (exn -> exn) -> (unit -> 'a t) -> ('a -> 'b t) -> (exn -> 'b t) -> 'b t *)
(* val backtrace_finalize : (exn -> exn) -> (unit -> 'a t) -> (unit -> unit t) -> 'a t *)
end
module Lwt_list : sig
# 1 "environment/v1/lwt_list.mli"
(* Lightweight thread library for OCaml
 * http://www.ocsigen.org/lwt
 * Interface Lwt_list
 * Copyright (C) 2010 Jérémie Dimino
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, with linking exceptions;
 * either version 2.1 of the License, or (at your option) any later
 * version. See COPYING file for details.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
 * 02111-1307, USA.
 *)

(** List helpers *)

(* TEZOS CHANGES

   * import version 2.4.5

*)

(** Note: this module use the same naming convention as
    {!Lwt_stream}. *)

(** {2 List iterators} *)

val iter_s : ('a -> unit Lwt.t) -> 'a list -> unit Lwt.t
val iter_p : ('a -> unit Lwt.t) -> 'a list -> unit Lwt.t

val iteri_s : (int -> 'a -> unit Lwt.t) -> 'a list -> unit Lwt.t
val iteri_p : (int -> 'a -> unit Lwt.t) -> 'a list -> unit Lwt.t

val map_s : ('a -> 'b Lwt.t) -> 'a list -> 'b list Lwt.t
val map_p : ('a -> 'b Lwt.t) -> 'a list -> 'b list Lwt.t

val mapi_s : (int -> 'a -> 'b Lwt.t) -> 'a list -> 'b list Lwt.t
val mapi_p : (int -> 'a -> 'b Lwt.t) -> 'a list -> 'b list Lwt.t

val rev_map_s : ('a -> 'b Lwt.t) -> 'a list -> 'b list Lwt.t
val rev_map_p : ('a -> 'b Lwt.t) -> 'a list -> 'b list Lwt.t

val fold_left_s : ('a -> 'b -> 'a Lwt.t) -> 'a -> 'b list -> 'a Lwt.t

val fold_right_s : ('a -> 'b -> 'b Lwt.t) -> 'a list -> 'b -> 'b Lwt.t

(** {2 List scanning} *)

val for_all_s : ('a -> bool Lwt.t) -> 'a list -> bool Lwt.t
val for_all_p : ('a -> bool Lwt.t) -> 'a list -> bool Lwt.t

val exists_s : ('a -> bool Lwt.t) -> 'a list -> bool Lwt.t
val exists_p : ('a -> bool Lwt.t) -> 'a list -> bool Lwt.t

(** {2 List searching} *)

val find_s : ('a -> bool Lwt.t) -> 'a list -> 'a Lwt.t

val filter_s : ('a -> bool Lwt.t) -> 'a list -> 'a list Lwt.t
val filter_p : ('a -> bool Lwt.t) -> 'a list -> 'a list Lwt.t

val filter_map_s : ('a -> 'b option Lwt.t) -> 'a list -> 'b list Lwt.t
val filter_map_p : ('a -> 'b option Lwt.t) -> 'a list -> 'b list Lwt.t

val partition_s : ('a -> bool Lwt.t) -> 'a list -> ('a list * 'a list) Lwt.t
val partition_p : ('a -> bool Lwt.t) -> 'a list -> ('a list * 'a list) Lwt.t
end
module MBytes : sig
# 1 "environment/v1/mBytes.mli"

type t

val create: int -> t

val length: t -> int

val copy: t -> t

val sub: t -> int -> int -> t
(** [sub src ofs len] extract a sub-array of [src] starting at [ofs]
    and of length [len]. No copying of elements is involved: the
    sub-array and the original array share the same storage space. *)

val shift: t -> int -> t
(** [shift src ofs] is equivalent to [sub src ofs (length src - ofs)] *)

val blit: t -> int -> t -> int -> int -> unit
(** [blit src ofs_src dst ofs_dst len] copy [len] bytes from [src]
    starting at [ofs_src] into [dst] starting at [ofs_dst].] *)

val blit_from_string: string -> int -> t -> int -> int -> unit
(** See [blit] *)

val blit_to_bytes: t -> int -> bytes -> int -> int -> unit
(** See [blit] *)

val of_string: string -> t
(** [of_string s] create an byte array filled with the same content than [s]. *)

val to_string: t -> string
(** [to_string b] dump the array content in a [string]. *)

val substring: t -> int -> int -> string
(** [substring b ofs len] is equivalent to [to_string (sub b ofs len)]. *)



(** Functions reading and writing bytes  *)

val get_char: t -> int -> char
(** [get_char buff i] reads 1 byte at offset i as a char *)

val get_uint8: t -> int -> int
(** [get_uint8 buff i] reads 1 byte at offset i as an unsigned int of 8
    bits. i.e. It returns a value between 0 and 2^8-1 *)

val get_int8: t -> int -> int
(** [get_int8 buff i] reads 1 byte at offset i as a signed int of 8
    bits. i.e. It returns a value between -2^7 and 2^7-1 *)

val set_char: t -> int -> char -> unit
(** [set_char buff i v] writes [v] to [buff] at offset [i] *)

val set_int8: t -> int -> int -> unit
(** [set_int8 buff i v] writes the least significant 8 bits of [v]
    to [buff] at offset [i] *)

(** Functions reading according to Big Endian byte order *)

val get_uint16: t -> int -> int
(** [get_uint16 buff i] reads 2 bytes at offset i as an unsigned int
      of 16 bits. i.e. It returns a value between 0 and 2^16-1 *)

val get_int16: t -> int -> int
(** [get_int16 buff i] reads 2 byte at offset i as a signed int of
      16 bits. i.e. It returns a value between -2^15 and 2^15-1 *)

val get_int32: t -> int -> int32
(** [get_int32 buff i] reads 4 bytes at offset i as an int32. *)

val get_int64: t -> int -> int64
(** [get_int64 buff i] reads 8 bytes at offset i as an int64. *)

val set_int16: t -> int -> int -> unit
(** [set_int16 buff i v] writes the least significant 16 bits of [v]
      to [buff] at offset [i] *)

val set_int32: t -> int -> int32 -> unit
(** [set_int32 buff i v] writes [v] to [buff] at offset [i] *)

val set_int64: t -> int -> int64 -> unit
(** [set_int64 buff i v] writes [v] to [buff] at offset [i] *)


module LE: sig

  (** Functions reading according to Little Endian byte order *)

  val get_uint16: t -> int -> int
  (** [get_uint16 buff i] reads 2 bytes at offset i as an unsigned int
      of 16 bits. i.e. It returns a value between 0 and 2^16-1 *)

  val get_int16: t -> int -> int
  (** [get_int16 buff i] reads 2 byte at offset i as a signed int of
      16 bits. i.e. It returns a value between -2^15 and 2^15-1 *)

  val get_int32: t -> int -> int32
  (** [get_int32 buff i] reads 4 bytes at offset i as an int32. *)

  val get_int64: t -> int -> int64
  (** [get_int64 buff i] reads 8 bytes at offset i as an int64. *)

  val set_int16: t -> int -> int -> unit
  (** [set_int16 buff i v] writes the least significant 16 bits of [v]
      to [buff] at offset [i] *)

  val set_int32: t -> int -> int32 -> unit
  (** [set_int32 buff i v] writes [v] to [buff] at offset [i] *)

  val set_int64: t -> int -> int64 -> unit
  (** [set_int64 buff i v] writes [v] to [buff] at offset [i] *)

end

val (=) : t -> t -> bool
val (<>) : t -> t -> bool
val (<) : t -> t -> bool
val (<=) : t -> t -> bool
val (>=) : t -> t -> bool
val (>) : t -> t -> bool
val compare : t -> t -> int

val concat: t -> t -> t
end
module Hex_encode : sig
# 1 "environment/v1/hex_encode.mli"
(** Tezos Utility library - Hexadecimal encoding *)

(** Parses a sequence of hexadecimal characters pairs as bytes *)
val hex_of_bytes: MBytes.t -> string

(** Prints a sequence of bytes as hexadecimal characters pairs *)
val bytes_of_hex: string -> MBytes.t

(** Interprets a sequence of hexadecimal characters pairs representing
    bytes as the characters codes of an OCaml string. *)
val hex_decode: string -> string

(** Formats the codes of the characters of an OCaml string as a
    sequence of hexadecimal character pairs. *)
val hex_encode: string -> string
end
module Uri : sig
# 1 "environment/v1/uri.mli"
type t
end
module Data_encoding : sig
# 1 "environment/v1/data_encoding.mli"

(** In memory JSON data *)
type json =
  [ `O of (string * json) list
  | `Bool of bool
  | `Float of float
  | `A of json list
  | `Null
  | `String of string ]

type json_schema

exception No_case_matched
exception Unexpected_tag of int
exception Duplicated_tag of int
exception Invalid_tag of int * [ `Uint8 | `Uint16 ]
exception Unexpected_enum of string * string list

type 'a t
type 'a encoding = 'a t

val classify : 'a encoding -> [ `Fixed of int | `Dynamic | `Variable ]

val splitted : json:'a encoding -> binary:'a encoding -> 'a encoding

val null : unit encoding
val empty : unit encoding
val unit : unit encoding
val constant : string -> unit encoding
val int8 : int encoding
val uint8 : int encoding
val int16 : int encoding
val uint16 : int encoding
val int31 : int encoding
val int32 : int32 encoding
val int64 : int64 encoding
val bool : bool encoding
val string : string encoding
val bytes : MBytes.t encoding
val float : float encoding
val option : 'a encoding -> 'a option encoding
val string_enum : (string * 'a) list -> 'a encoding

module Fixed : sig
  val string : int -> string encoding
  val bytes : int -> MBytes.t encoding
end

module Variable : sig
  val string : string encoding
  val bytes : MBytes.t encoding
  val array : 'a encoding -> 'a array encoding
  val list : 'a encoding -> 'a list encoding
end

val dynamic_size : 'a encoding -> 'a encoding

val json : json encoding
val json_schema : json_schema encoding

type 'a field
val req :
  ?title:string -> ?description:string ->
  string -> 't encoding -> 't field
val opt :
  ?title:string -> ?description:string ->
  string -> 't encoding -> 't option field
val varopt :
  ?title:string -> ?description:string ->
  string -> 't encoding -> 't option field
val dft :
  ?title:string -> ?description:string ->
  string -> 't encoding -> 't -> 't field

val obj1 :
  'f1 field -> 'f1 encoding
val obj2 :
  'f1 field -> 'f2 field -> ('f1 * 'f2) encoding
val obj3 :
  'f1 field -> 'f2 field -> 'f3 field -> ('f1 * 'f2 * 'f3) encoding
val obj4 :
  'f1 field -> 'f2 field -> 'f3 field -> 'f4 field ->
  ('f1 * 'f2 * 'f3 * 'f4) encoding
val obj5 :
  'f1 field -> 'f2 field -> 'f3 field -> 'f4 field -> 'f5 field ->
  ('f1 * 'f2 * 'f3 * 'f4 * 'f5) encoding
val obj6 :
  'f1 field -> 'f2 field -> 'f3 field -> 'f4 field -> 'f5 field ->
  'f6 field ->
  ('f1 * 'f2 * 'f3 * 'f4 * 'f5 * 'f6) encoding
val obj7 :
  'f1 field -> 'f2 field -> 'f3 field -> 'f4 field -> 'f5 field ->
  'f6 field -> 'f7 field ->
  ('f1 * 'f2 * 'f3 * 'f4 * 'f5 * 'f6 * 'f7) encoding
val obj8 :
  'f1 field -> 'f2 field -> 'f3 field -> 'f4 field -> 'f5 field ->
  'f6 field -> 'f7 field -> 'f8 field ->
  ('f1 * 'f2 * 'f3 * 'f4 * 'f5 * 'f6 * 'f7 * 'f8) encoding
val obj9 :
  'f1 field -> 'f2 field -> 'f3 field -> 'f4 field -> 'f5 field ->
  'f6 field -> 'f7 field -> 'f8 field -> 'f9 field ->
  ('f1 * 'f2 * 'f3 * 'f4 * 'f5 * 'f6 * 'f7 * 'f8 * 'f9) encoding
val obj10 :
  'f1 field -> 'f2 field -> 'f3 field -> 'f4 field -> 'f5 field ->
  'f6 field -> 'f7 field -> 'f8 field -> 'f9 field -> 'f10 field ->
  ('f1 * 'f2 * 'f3 * 'f4 * 'f5 * 'f6 * 'f7 * 'f8 * 'f9 * 'f10) encoding

val tup1 :
  'f1 encoding ->
  'f1 encoding
val tup2 :
  'f1 encoding -> 'f2 encoding ->
  ('f1 * 'f2) encoding
val tup3 :
  'f1 encoding -> 'f2 encoding -> 'f3 encoding ->
  ('f1 * 'f2 * 'f3) encoding
val tup4 :
  'f1 encoding -> 'f2 encoding -> 'f3 encoding -> 'f4 encoding ->
  ('f1 * 'f2 * 'f3 * 'f4) encoding
val tup5 :
  'f1 encoding -> 'f2 encoding -> 'f3 encoding -> 'f4 encoding ->
  'f5 encoding ->
  ('f1 * 'f2 * 'f3 * 'f4 * 'f5) encoding
val tup6 :
  'f1 encoding -> 'f2 encoding -> 'f3 encoding -> 'f4 encoding ->
  'f5 encoding -> 'f6 encoding ->
  ('f1 * 'f2 * 'f3 * 'f4 * 'f5 * 'f6) encoding
val tup7 :
  'f1 encoding -> 'f2 encoding -> 'f3 encoding -> 'f4 encoding ->
  'f5 encoding -> 'f6 encoding -> 'f7 encoding ->
  ('f1 * 'f2 * 'f3 * 'f4 * 'f5 * 'f6 * 'f7) encoding
val tup8 :
  'f1 encoding -> 'f2 encoding -> 'f3 encoding -> 'f4 encoding ->
  'f5 encoding -> 'f6 encoding -> 'f7 encoding -> 'f8 encoding ->
  ('f1 * 'f2 * 'f3 * 'f4 * 'f5 * 'f6 * 'f7 * 'f8) encoding
val tup9 :
  'f1 encoding -> 'f2 encoding -> 'f3 encoding -> 'f4 encoding ->
  'f5 encoding -> 'f6 encoding -> 'f7 encoding -> 'f8 encoding ->
  'f9 encoding ->
  ('f1 * 'f2 * 'f3 * 'f4 * 'f5 * 'f6 * 'f7 * 'f8 * 'f9) encoding
val tup10 :
  'f1 encoding -> 'f2 encoding -> 'f3 encoding -> 'f4 encoding ->
  'f5 encoding -> 'f6 encoding -> 'f7 encoding -> 'f8 encoding ->
  'f9 encoding -> 'f10 encoding ->
  ('f1 * 'f2 * 'f3 * 'f4 * 'f5 * 'f6 * 'f7 * 'f8 * 'f9 * 'f10) encoding

val merge_objs : 'o1 encoding -> 'o2 encoding -> ('o1 * 'o2) encoding
val merge_tups : 'a1 encoding -> 'a2 encoding -> ('a1 * 'a2) encoding

val array : 'a encoding -> 'a array encoding
val list : 'a encoding -> 'a list encoding

val assoc : 'a encoding -> (string * 'a) list encoding

type 't case
val case :
  ?tag:int -> 'a encoding -> ('t -> 'a option) -> ('a -> 't) -> 't case
val union :
  ?tag_size:[ `Uint8 | `Uint16 ] -> 't case list -> 't encoding

val describe :
  ?title:string -> ?description:string ->
  't encoding ->'t encoding

val def : string -> 'a encoding -> 'a encoding

val conv :
  ('a -> 'b) -> ('b -> 'a) ->
  ?schema:json_schema ->
  'b encoding -> 'a encoding

val mu : string -> ('a encoding -> 'a encoding) -> 'a encoding

module Json : sig

  val schema : 'a encoding -> json_schema
  val construct : 't encoding -> 't -> json
  val destruct : 't encoding -> json -> 't

  (** JSON Error *)

  type path = path_item list
  and path_item =
    [ `Field of string
    (** A field in an object. *)
    | `Index of int
    (** An index in an array. *)
    | `Star
    (** Any / every field or index. *)
    | `Next
      (** The next element after an array. *) ]

  (** Exception raised by destructors, with the location in the original
      JSON structure and the specific error. *)
  exception Cannot_destruct of (path * exn)

  (** Unexpected kind of data encountered (w/ the expectation). *)
  exception Unexpected of string * string

  (** Some {!union} couldn't be destructed, w/ the reasons for each {!case}. *)
  exception No_case_matched of exn list

  (** Array of unexpected size encountered  (w/ the expectation). *)
  exception Bad_array_size of int * int

  (** Missing field in an object. *)
  exception Missing_field of string

  (** Supernumerary field in an object. *)
  exception Unexpected_field of string

  val print_error :
    ?print_unknown: (Format.formatter -> exn -> unit) ->
    Format.formatter -> exn -> unit

  (** Helpers for writing encoders. *)
  val cannot_destruct : ('a, Format.formatter, unit, 'b) format4 -> 'a
  val wrap_error : ('a -> 'b) -> 'a -> 'b

end

module Binary : sig

  val length : 'a encoding -> 'a -> int
  val fixed_length : 'a encoding -> int option
  val read : 'a encoding -> MBytes.t -> int -> int -> (int * 'a) option
  val write : 'a encoding -> 'a -> MBytes.t -> int -> int option
  val to_bytes : 'a encoding -> 'a -> MBytes.t
  val of_bytes : 'a encoding -> MBytes.t -> 'a option

end
end
module Error_monad : sig
# 1 "environment/v1/error_monad.mli"
(** Tezos Protocol Implementation - Error Monad *)

(** {2 Error classification} *************************************************)

(** Categories of error *)
type error_category =
  [ `Branch (** Errors that may not happen in another context *)
  | `Temporary (** Errors that may not happen in a later context *)
  | `Permanent (** Errors that will happen no matter the context *)
  ]

(** Custom error handling for economic protocols. *)

type error = ..

val pp : Format.formatter -> error -> unit

(** A JSON error serializer *)
val error_encoding : unit -> error Data_encoding.t
val json_of_error : error -> Data_encoding.json
val error_of_json : Data_encoding.json -> error

(** For other modules to register specialized error serializers *)
val register_error_kind :
  error_category ->
  id:string -> title:string -> description:string ->
  ?pp:(Format.formatter -> 'err -> unit) ->
  'err Data_encoding.t ->
  (error -> 'err option) -> ('err -> error) ->
  unit

(** Classify an error using the registered kinds *)
val classify_errors : error list -> error_category

(** {2 Monad definition} *****************************************************)

(** The error monad wrapper type, the error case holds a stack of
    error, initialized by the first call to {!fail} and completed by
    each call to {!trace} as the stack is rewinded. The most general
    error is thus at the top of the error stack, going down to the
    specific error that actually caused the failure. *)
type 'a tzresult = ('a, error list) result

(** A JSON serializer for result of a given type *)
val result_encoding : 'a Data_encoding.t -> 'a tzresult Data_encoding.encoding

(** Sucessful result *)
val ok : 'a -> 'a tzresult

(** Sucessful return *)
val return : 'a -> 'a tzresult Lwt.t

(** Erroneous result *)
val error : error -> 'a tzresult

(** Erroneous return *)
val fail : error -> 'a tzresult Lwt.t

(** Non-Lwt bind operator *)
val (>>?) : 'a tzresult -> ('a -> 'b tzresult) -> 'b tzresult

(** Bind operator *)
val (>>=?) : 'a tzresult Lwt.t -> ('a -> 'b tzresult Lwt.t) -> 'b tzresult Lwt.t

(** Lwt's bind reexported *)
val (>>=) : 'a Lwt.t -> ('a -> 'b Lwt.t) -> 'b Lwt.t
val (>|=) : 'a Lwt.t -> ('a -> 'b) -> 'b Lwt.t

(** To operator *)
val (>>|?) : 'a tzresult Lwt.t -> ('a -> 'b) -> 'b tzresult Lwt.t

(** Non-Lwt to operator *)
val (>|?) : 'a tzresult -> ('a -> 'b) -> 'b tzresult

(** Enrich an error report (or do nothing on a successful result) manually *)
val record_trace : error -> 'a tzresult -> 'a tzresult

(** Automatically enrich error reporting on stack rewind *)
val trace : error -> 'b tzresult Lwt.t -> 'b tzresult Lwt.t

(** Erroneous return on failed assertion *)
val fail_unless : bool -> error -> unit tzresult Lwt.t

(** {2 In-monad list iterators} **********************************************)

(** A {!List.iter} in the monad *)
val iter_s : ('a -> unit tzresult Lwt.t) -> 'a list -> unit tzresult Lwt.t

(** A {!List.map} in the monad *)
val map_s : ('a -> 'b tzresult Lwt.t) -> 'a list -> 'b list tzresult Lwt.t
val map_p : ('a -> 'b tzresult Lwt.t) -> 'a list -> 'b list tzresult Lwt.t

(** A {!List.map2} in the monad *)
val map2 :
  ('a -> 'b -> 'c tzresult) -> 'a list -> 'b list -> 'c list tzresult

(** A {!List.map2} in the monad *)
val map2_s :
  ('a -> 'b -> 'c tzresult Lwt.t) -> 'a list -> 'b list ->
  'c list tzresult Lwt.t

(** A {!List.filter_map} in the monad *)
val filter_map_s : ('a -> 'b option tzresult Lwt.t) -> 'a list -> 'b list tzresult Lwt.t

(** A {!List.fold_left} in the monad *)
val fold_left_s : ('a -> 'b -> 'a tzresult Lwt.t) -> 'a -> 'b list -> 'a tzresult Lwt.t

(** A {!List.fold_right} in the monad *)
val fold_right_s : ('a -> 'b -> 'b tzresult Lwt.t) -> 'a list -> 'b -> 'b tzresult Lwt.t
end
open Error_monad
module Logging : sig
# 1 "environment/v1/logging.mli"

val debug: ('a, Format.formatter, unit, unit) format4 -> 'a
val log_info: ('a, Format.formatter, unit, unit) format4 -> 'a
val log_notice: ('a, Format.formatter, unit, unit) format4 -> 'a
val warn: ('a, Format.formatter, unit, unit) format4 -> 'a
val log_error: ('a, Format.formatter, unit, unit) format4 -> 'a
val fatal_error: ('a, Format.formatter, unit, 'b) format4 -> 'a

val lwt_debug: ('a, Format.formatter, unit, unit Lwt.t) format4 -> 'a
val lwt_log_info: ('a, Format.formatter, unit, unit Lwt.t) format4 -> 'a
val lwt_log_notice: ('a, Format.formatter, unit, unit Lwt.t) format4 -> 'a
val lwt_warn: ('a, Format.formatter, unit, unit Lwt.t) format4 -> 'a
val lwt_log_error: ('a, Format.formatter, unit, unit Lwt.t) format4 -> 'a
end
module Time : sig
# 1 "environment/v1/time.mli"

type t

val add : t -> int64 -> t
val diff : t -> t -> int64

val equal : t -> t -> bool
val compare : t -> t -> int

val (=) : t -> t -> bool
val (<>) : t -> t -> bool
val (<) : t -> t -> bool
val (<=) : t -> t -> bool
val (>=) : t -> t -> bool
val (>) : t -> t -> bool
val min : t -> t -> t
val max : t -> t -> t

val of_seconds : int64 -> t
val to_seconds : t -> int64

val of_notation : string -> t option
val of_notation_exn : string -> t
val to_notation : t -> string

val encoding : t Data_encoding.t
val rfc_encoding : t Data_encoding.t

val pp_hum : Format.formatter -> t -> unit



end
module Base58 : sig
# 1 "environment/v1/base58.mli"

type 'a encoding

val simple_decode: 'a encoding -> string -> 'a option
val simple_encode: 'a encoding -> 'a -> string

type data = ..

val register_encoding:
  prefix: string ->
  length: int ->
  to_raw: ('a -> string) ->
  of_raw: (string -> 'a option) ->
  wrap: ('a -> data) ->
  'a encoding

val check_encoded_prefix: 'a encoding -> string -> int -> unit

val decode: string -> data option
end
module Hash : sig
# 1 "environment/v1/hash.mli"

(** Tezos - Manipulation and creation of hashes *)

(** {2 Hash Types} ************************************************************)

(** The signature of an abstract hash type, as produced by functor
    {!Make_SHA256}. The {!t} type is abstracted for separating the
    various kinds of hashes in the system at typing time. Each type is
    equipped with functions to use it as is of as keys in the database
    or in memory sets and maps. *)

module type MINIMAL_HASH = sig

  type t

  val name: string
  val title: string

  val hash_bytes: MBytes.t list -> t
  val hash_string: string list -> t
  val size: int (* in bytes *)
  val compare: t -> t -> int
  val equal: t -> t -> bool

  val to_hex: t -> string
  val of_hex: string -> t option
  val of_hex_exn: string -> t

  val to_string: t -> string
  val of_string: string -> t option
  val of_string_exn: string -> t

  val to_bytes: t -> MBytes.t
  val of_bytes: MBytes.t -> t option
  val of_bytes_exn: MBytes.t -> t

  val read: MBytes.t -> int -> t
  val write: MBytes.t -> int -> t -> unit

  val to_path: t -> string list
  val of_path: string list -> t option
  val of_path_exn: string list -> t

  val prefix_path: string -> string list
  val path_length: int

end

module type HASH = sig

  include MINIMAL_HASH

  val of_b58check_exn: string -> t
  val of_b58check_opt: string -> t option
  val to_b58check: t -> string
  val to_short_b58check: t -> string
  val encoding: t Data_encoding.t
  val pp: Format.formatter -> t -> unit
  val pp_short: Format.formatter -> t -> unit
  type Base58.data += Hash of t
  val b58check_encoding: t Base58.encoding

  module Set : sig
    include Set.S with type elt = t
    val encoding: t Data_encoding.t
  end

  module Map : sig
    include Map.S with type key = t
    val encoding: 'a Data_encoding.t -> 'a t Data_encoding.t
  end

end

module type MERKLE_TREE = sig
  type elt
  include HASH
  val compute: elt list -> t
  val empty: t
  type path =
    | Left of path * t
    | Right of t * path
    | Op
  val compute_path: elt list -> int -> path
  val check_path: path -> elt -> t * int
  val path_encoding: path Data_encoding.t
end

(** {2 Building Hashes} *******************************************************)

(** The parameters for creating a new Hash type using
    {!Make_Blake2B}. Both {!name} and {!title} are only informative,
    used in error messages and serializers. *)

module type Name = sig
  val name : string
  val title : string
  val size : int option
end

module type PrefixedName = sig
  include Name
  val b58check_prefix : string
end

(** Builds a new Hash type using Sha256. *)

module Make_minimal_Blake2B (Name : Name) : MINIMAL_HASH
module Make_Blake2B
    (Register : sig
       val register_encoding:
         prefix: string ->
         length: int ->
         to_raw: ('a -> string) ->
         of_raw: (string -> 'a option) ->
         wrap: ('a -> Base58.data) ->
         'a Base58.encoding
     end)
    (Name : PrefixedName) : HASH

(** {2 Predefined Hashes } ****************************************************)

(** Blocks hashes / IDs. *)
module Block_hash : HASH

(** Operations hashes / IDs. *)
module Operation_hash : HASH

(** List of operations hashes / IDs. *)
module Operation_list_hash :
  MERKLE_TREE with type elt = Operation_hash.t

module Operation_list_list_hash :
  MERKLE_TREE with type elt = Operation_list_hash.t

(** Protocol versions / source hashes. *)
module Protocol_hash : HASH

module Net_id : HASH
end
open Hash
module Ed25519 : sig
# 1 "environment/v1/ed25519.mli"
(** Tezos - Ed25519 cryptography *)


(** {2 Hashed public keys for user ID} ***************************************)

module Public_key_hash : Hash.HASH


(** {2 Signature} ************************************************************)

module Public_key : sig

  include Compare.S
  val encoding: t Data_encoding.t

  val hash: t -> Public_key_hash.t

  type Base58.data +=
    | Public_key of t

  val of_b58check_exn: string -> t
  val of_b58check_opt: string -> t option
  val to_b58check: t -> string

  val of_bytes: Bytes.t -> t

end

module Secret_key : sig

  type t
  val encoding: t Data_encoding.t

  type Base58.data +=
    | Secret_key of t

  val of_b58check_exn: string -> t
  val of_b58check_opt: string -> t option
  val to_b58check: t -> string

  val of_bytes: Bytes.t -> t

end

module Signature : sig

  type t
  val encoding: t Data_encoding.t

  type Base58.data +=
    | Signature of t

  val of_b58check_exn: string -> t
  val of_b58check_opt: string -> t option
  val to_b58check: t -> string

  val of_bytes: Bytes.t -> t

  (** Checks a signature *)
  val check: Public_key.t -> t -> MBytes.t -> bool

  (** Append a signature *)
  val append: Secret_key.t -> MBytes.t -> MBytes.t

end

val sign: Secret_key.t -> MBytes.t -> Signature.t

val generate_key: unit -> (Public_key_hash.t * Public_key.t * Secret_key.t)
end
module Tezos_data : sig
# 1 "environment/v1/tezos_data.mli"
(**************************************************************************)
(*                                                                        *)
(*    Copyright (c) 2014 - 2016.                                          *)
(*    Dynamic Ledger Solutions, Inc. <contact@tezos.com>                  *)
(*                                                                        *)
(*    All rights reserved. No warranty, explicit or implicit, provided.   *)
(*                                                                        *)
(**************************************************************************)

module type DATA = sig

  type t

  val compare: t -> t -> int
  val equal: t -> t -> bool

  val pp: Format.formatter -> t -> unit

  val encoding: t Data_encoding.t
  val to_bytes: t -> MBytes.t
  val of_bytes: MBytes.t -> t option

end

module Fitness : DATA with type t = MBytes.t list

module type HASHABLE_DATA = sig

  include DATA

  type hash
  val hash: t -> hash
  val hash_raw: MBytes.t -> hash

end

module Operation : sig

  type shell_header = {
    net_id: Net_id.t ;
    branch: Block_hash.t ;
  }
  val shell_header_encoding: shell_header Data_encoding.t

  type t = {
    shell: shell_header ;
    proto: MBytes.t ;
  }

  include HASHABLE_DATA with type t := t
                         and type hash := Operation_hash.t

end

module Block_header : sig

  type shell_header = {
    net_id: Net_id.t ;
    level: Int32.t ;
    proto_level: int ; (* uint8 *)
    predecessor: Block_hash.t ;
    timestamp: Time.t ;
    operations_hash: Operation_list_list_hash.t ;
    fitness: MBytes.t list ;
  }

  val shell_header_encoding: shell_header Data_encoding.t

  type t = {
    shell: shell_header ;
    proto: MBytes.t ;
  }

  include HASHABLE_DATA with type t := t
                         and type hash := Block_hash.t

end

module Protocol : sig

  type t = {
    expected_env: env_version ;
    components: component list ;
  }

  (** An OCaml source component of a protocol implementation. *)
  and component = {
    (** The OCaml module name. *)
    name : string ;
    (** The OCaml interface source code *)
    interface : string option ;
    (** The OCaml source code *)
    implementation : string ;
  }

  and env_version = V1

  val component_encoding: component Data_encoding.t
  val env_version_encoding: env_version Data_encoding.t

  include HASHABLE_DATA with type t := t
                         and type hash := Protocol_hash.t

end
end
open Tezos_data
module Persist : sig
# 1 "environment/v1/persist.mli"
(**  Tezos - Persistent structures on top of {!Store} or {!Context} *)

(** Keys in (kex x value) database implementations *)
type key = string list

(** Values in (kex x value) database implementations *)
type value = MBytes.t

(** Low level view over a (key x value) database implementation. *)
module type STORE = sig
  type t
  val mem: t -> key -> bool Lwt.t
  val dir_mem: t -> key -> bool Lwt.t
  val get: t -> key -> value option Lwt.t
  val set: t -> key -> value -> t Lwt.t
  val del: t -> key -> t Lwt.t
  val list: t -> key list -> key list Lwt.t
  val remove_rec: t -> key -> t Lwt.t
end

(** Projection of OCaml keys of some abstract type to concrete storage
    keys. For practical reasons, all such keys must fall under a same
    {!prefix} and have the same relative {!length}. Functions
    {!to_path} and {!of_path} only take the relative part into account
    (the prefix is added and removed when needed). *)
module type KEY = sig
  type t
  val prefix: key
  val length: int
  val to_path: t -> key
  val of_path: key -> t
  val compare: t -> t -> int
end

(** A KEY instance for using raw implementation paths as keys *)
module RawKey : KEY with type t = key

(** Projection of OCaml values of some abstract type to concrete
    storage data. *)
module type VALUE = sig
  type t
  val of_bytes: value -> t option
  val to_bytes: t -> value
end

(** A VALUE instance for using the raw bytes values *)
module RawValue : VALUE with type t = value

module type BYTES_STORE = sig
  type t
  type key
  val mem: t -> key -> bool Lwt.t
  val get: t -> key -> value option Lwt.t
  val set: t -> key -> value -> t Lwt.t
  val del: t -> key -> t Lwt.t
  val list: t -> key list -> key list Lwt.t
  val remove_rec: t -> key -> t Lwt.t
end

module MakeBytesStore (S : STORE) (K : KEY) :
  BYTES_STORE with type t = S.t and type key = K.t

(** {2 Typed Store Overlays} *************************************************)

(** Signature of a typed store as returned by {!MakecoTypedStore} *)
module type TYPED_STORE = sig
  type t
  type key
  type value
  val mem: t -> key -> bool Lwt.t
  val get: t -> key -> value option Lwt.t
  val set: t -> key -> value -> t Lwt.t
  val del: t -> key -> t Lwt.t
end

(** Gives a typed view of a store (values of a given type stored under
    keys of a given type). The view is also restricted to a prefix,
    (which can be empty). For all primitives to work as expected, all
    keys under this prefix must be homogeneously typed. *)
module MakeTypedStore (S : STORE) (K : KEY) (C : VALUE) :
  TYPED_STORE with type t = S.t and type key = K.t and type value = C.t

(** {2 Persistent Sets} ******************************************************)

(** Signature of a set as returned by {!MakePersistentSet} *)
module type PERSISTENT_SET = sig
  type t and key
  val mem : t -> key -> bool Lwt.t
  val set : t -> key -> t Lwt.t
  val del : t -> key -> t Lwt.t
  val elements : t -> key list Lwt.t
  val clear : t -> t Lwt.t
  val iter : t -> f:(key -> unit Lwt.t) -> unit Lwt.t
  val fold : t -> 'a -> f:(key -> 'a -> 'a Lwt.t) -> 'a Lwt.t
end

(** Signature of a buffered set as returned by {!MakeBufferedPersistentSet} *)
module type BUFFERED_PERSISTENT_SET = sig
  include PERSISTENT_SET
  module Set : Set.S with type elt = key
  val read : t -> Set.t Lwt.t
  val write : t -> Set.t -> t Lwt.t
end

(** Build a set in the (key x value) storage by encoding elements as
    keys and using the association of (any) data to these keys as
    membership. For this to work, the prefix passed must be reserved
    for the set (every key under it is considered a member). *)
module MakePersistentSet (S : STORE) (K : KEY)
  : PERSISTENT_SET with type t := S.t and type key := K.t

(** Same as {!MakePersistentSet} but also provides a way to use an
    OCaml set as an explicitly synchronized in-memory buffer. *)
module MakeBufferedPersistentSet
    (S : STORE) (K : KEY) (Set : Set.S with type elt = K.t)
  : BUFFERED_PERSISTENT_SET
    with type t := S.t
     and type key := K.t
     and module Set := Set

(** {2 Persistent Maps} ******************************************************)

(** Signature of a map as returned by {!MakePersistentMap} *)
module type PERSISTENT_MAP = sig
  type t and key and value
  val mem : t -> key -> bool Lwt.t
  val get : t -> key -> value option Lwt.t
  val set : t -> key -> value -> t Lwt.t
  val del : t -> key -> t Lwt.t
  val bindings : t -> (key * value) list Lwt.t
  val clear : t -> t Lwt.t
  val iter : t -> f:(key -> value -> unit Lwt.t) -> unit Lwt.t
  val fold : t -> 'a -> f:(key -> value -> 'a -> 'a Lwt.t) -> 'a Lwt.t
end

(** Signature of a buffered map as returned by {!MakeBufferedPersistentMap} *)
module type BUFFERED_PERSISTENT_MAP = sig
  include PERSISTENT_MAP
  module Map : Map.S with type key = key
  val read : t -> value Map.t Lwt.t
  val write : t -> value Map.t -> t Lwt.t
end

(** Build a map in the (key x value) storage. For this to work, the
    prefix passed must be reserved for the map (every key under it is
    considered the key of a binding). *)
module MakePersistentMap (S : STORE) (K : KEY) (C : VALUE)
  : PERSISTENT_MAP
    with type t := S.t and type key := K.t and type value := C.t

(** Same as {!MakePersistentMap} but also provides a way to use an
    OCaml map as an explicitly synchronized in-memory buffer. *)
module MakeBufferedPersistentMap
    (S : STORE) (K : KEY) (C : VALUE) (Map : Map.S with type key = K.t)
 : BUFFERED_PERSISTENT_MAP
   with type t := S.t
    and type key := K.t
    and type value := C.t
    and module Map := Map

(** {2 Predefined Instances} *************************************************)

module MakePersistentBytesMap (S : STORE) (K : KEY)
  : PERSISTENT_MAP
  with type t := S.t and type key := K.t and type value := MBytes.t

module MakeBufferedPersistentBytesMap
    (S : STORE) (K : KEY) (Map : Map.S with type key = K.t)
  : BUFFERED_PERSISTENT_MAP
    with type t := S.t
     and type key := K.t
     and type value := MBytes.t
     and module Map := Map

module type TYPED_VALUE_REPR = sig
  type value
  val encoding: value Data_encoding.t
end

module MakePersistentTypedMap (S : STORE) (K : KEY) (T : TYPED_VALUE_REPR)
  : PERSISTENT_MAP
    with type t := S.t and type key := K.t and type value := T.value

module MakeBufferedPersistentTypedMap
    (S : STORE) (K : KEY) (T : TYPED_VALUE_REPR) (Map : Map.S with type key = K.t)
  : BUFFERED_PERSISTENT_MAP
    with type t := S.t
     and type key := K.t
     and type value := T.value
     and module Map := Map

module MakeHashResolver
    (Store : sig
       type t
       val dir_mem: t -> key -> bool Lwt.t
       val list: t -> key list -> key list Lwt.t
       val prefix: string list
     end)
    (H: Hash.HASH) : sig
  val resolve : Store.t -> string -> H.t list Lwt.t
end
end
module Context : sig
# 1 "environment/v1/context.mli"
(** View over the context store, restricted to types, access and
    functional manipulation of an existing context. *)

include Persist.STORE

val register_resolver:
  'a Base58.encoding -> (t -> string -> 'a list Lwt.t) -> unit

val complete: t -> string -> string list Lwt.t
end
module RPC : sig
# 1 "environment/v1/RPC.mli"
(** View over the RPC service, restricted to types. A protocol
    implementation can define a set of remote procedures which are
    registered when the protocol is activated via its [rpcs]
    function. However, it cannot register new or update existing
    procedures afterwards, neither can it see other procedures. *)

(** Typed path argument. *)
module Arg : sig

  type 'a arg
  val make:
    ?descr:string ->
    name:string ->
    destruct:(string -> ('a, string) result) ->
    construct:('a -> string) ->
    unit -> 'a arg

  type descr = {
    name: string ;
    descr: string option ;
  }
  val descr: 'a arg -> descr

  val int: int arg
  val int32: int32 arg
  val int64: int64 arg
  val float: float arg

end

(** Parametrized path to services. *)
module Path : sig

  type ('prefix, 'params) path
  type 'prefix context = ('prefix, 'prefix) path

  val root: 'a context

  val add_suffix:
    ('prefix, 'params) path -> string -> ('prefix, 'params) path
  val (/):
    ('prefix, 'params) path -> string -> ('prefix, 'params) path

  val add_arg:
    ('prefix, 'params) path -> 'a Arg.arg -> ('prefix, 'params * 'a) path
  val (/:):
    ('prefix, 'params) path -> 'a Arg.arg -> ('prefix, 'params * 'a) path

  val prefix:
    ('prefix, 'a) path -> ('a, 'params) path -> ('prefix, 'params) path

  val map:
    ('a -> 'b) -> ('b -> 'a) -> ('prefix, 'a) path -> ('prefix, 'b) path

end

(** Services. *)
type ('prefix, 'params, 'input, 'output) service

(** HTTP methods as defined in Cohttp.Code *)
type meth = [
  | `GET
  | `POST
  | `HEAD
  | `DELETE
  | `PATCH
  | `PUT
  | `OPTIONS
  | `TRACE
  | `CONNECT
  | `Other of string
]

val service:
  ?meth: meth ->
  ?description: string ->
  input: 'input Data_encoding.t ->
  output: 'output Data_encoding.t ->
  ('prefix, 'params) Path.path ->
  ('prefix, 'params, 'input, 'output) service

module Answer : sig

  (** Return type for service handler *)
  type 'a answer =
    { code : int ;
      body : 'a output ;
    }

  and 'a output =
    | Empty
    | Single of 'a
    | Stream of 'a stream

  and 'a stream = {
    next: unit -> 'a option Lwt.t ;
    shutdown: unit -> unit ;
  }

  val ok: 'a -> 'a answer
  val answer: ?code:int -> 'a -> 'a answer
  val return: ?code:int -> 'a -> 'a answer Lwt.t

end

(** Dispatch tree *)
type 'prefix directory

val empty: 'prefix directory

(** Registring handler in service tree. *)
val register:
  'prefix directory ->
  ('prefix, 'params, 'input, 'output) service ->
  ('params -> 'input -> 'output Answer.answer Lwt.t) ->
  'prefix directory

(** Registring handler in service tree. Curryfied variant.  *)
val register0:
  unit directory ->
  (unit, unit, 'i, 'o) service ->
  ('i -> 'o Answer.answer Lwt.t) ->
  unit directory

val register1:
  'prefix directory ->
  ('prefix, unit * 'a, 'i, 'o) service ->
  ('a -> 'i -> 'o Answer.answer Lwt.t) ->
  'prefix directory

val register2:
  'prefix directory ->
  ('prefix, (unit * 'a) * 'b, 'i, 'o) service ->
  ('a -> 'b -> 'i -> 'o Answer.answer Lwt.t) ->
  'prefix directory

val register3:
  'prefix directory ->
  ('prefix, ((unit * 'a) * 'b) * 'c, 'i, 'o) service ->
  ('a -> 'b -> 'c -> 'i -> 'o Answer.answer Lwt.t) ->
  'prefix directory

val register4:
  'prefix directory ->
  ('prefix, (((unit * 'a) * 'b) * 'c) * 'd, 'i, 'o) service ->
  ('a -> 'b -> 'c -> 'd -> 'i -> 'o Answer.answer Lwt.t) ->
  'prefix directory

val register5:
  'prefix directory ->
  ('prefix, ((((unit * 'a) * 'b) * 'c) * 'd) * 'e, 'i, 'o) service ->
  ('a -> 'b -> 'c -> 'd -> 'e -> 'i -> 'o Answer.answer Lwt.t) ->
  'prefix directory
end
module Updater : sig
# 1 "environment/v1/updater.mli"
(** Tezos Protocol Environment - Protocol Implementation Updater *)

type validation_result = {
  context: Context.t ;
  fitness: Fitness.t ;
  message: string option ;
  max_operations_ttl: int ;
}

type rpc_context = {
  block_hash: Block_hash.t ;
  block_header: Block_header.t ;
  operation_hashes: unit -> Operation_hash.t list list Lwt.t ;
  operations: unit -> Operation.t list list Lwt.t ;
  context: Context.t ;
}

(** This is the signature of a Tezos protocol implementation. It has
    access to the standard library and the Environment module. *)
module type PROTOCOL = sig

  (** The version specific type of operations. *)
  type operation

  (** The maximum size of operations in bytes *)
  val max_operation_data_length: int

  (** The maximum size of block headers in bytes *)
  val max_block_length: int

  (** The maximum *)
  val max_number_of_operations: int

  (** The parsing / preliminary validation function for
      operations. Similar to {!parse_block}. *)
  val parse_operation:
    Operation_hash.t -> Operation.t -> operation tzresult

  (** Basic ordering of operations. [compare_operations op1 op2] means
      that [op1] should appear before [op2] in a block. *)
  val compare_operations: operation -> operation -> int

  (** A functional state that is transmitted through the steps of a
      block validation sequence. It must retain the current state of
      the store (that can be extracted from the outside using
      {!current_context}, and whose final value is produced by
      {!finalize_block}). It can also contain the information that
      must be remembered during the validation, which must be
      immutable (as validator or baker implementations are allowed to
      pause, replay or backtrack during the validation process). *)
  type validation_state

  (** Access the context at a given validation step. *)
  val current_context: validation_state -> Context.t tzresult Lwt.t

  (** Checks that a block is well formed in a given context. This
      function should run quickly, as its main use is to reject bad
      blocks from the network as early as possible. The input context
      is the one resulting of an ancestor block of same protocol
      version, not necessarily the one of its predecessor. *)
  val precheck_block:
    ancestor_context: Context.t ->
    ancestor_timestamp: Time.t ->
    Block_header.t ->
    unit tzresult Lwt.t

  (** The first step in a block validation sequence. Initializes a
      validation context for validating a block. Takes as argument the
      {!Block_header.t} to initialize the context for this block, patching
      the context resulting of the application of the predecessor
      block passed as parameter. The function {!precheck_block} may
      not have been called before [begin_application], so all the
      check performed by the former must be repeated in the latter. *)
  val begin_application:
    predecessor_context: Context.t ->
    predecessor_timestamp: Time.t ->
    predecessor_fitness: Fitness.t ->
    Block_header.t ->
    validation_state tzresult Lwt.t

  (** Initializes a validation context for constructing a new block
      (as opposed to validating an existing block). Since there is no
      {!Block_header.t} header available, the parts that it provides are
      passed as arguments (predecessor block hash, context resulting
      of the application of the predecessor block, and timestamp). *)
  val begin_construction:
    predecessor_context: Context.t ->
    predecessor_timestamp: Time.t ->
    predecessor_level: Int32.t ->
    predecessor_fitness: Fitness.t ->
    predecessor: Block_hash.t ->
    timestamp: Time.t ->
    ?proto_header: MBytes.t ->
    unit -> validation_state tzresult Lwt.t

  (** Called after {!begin_application} (or {!begin_construction}) and
      before {!finalize_block}, with each operation in the block. *)
  val apply_operation:
    validation_state -> operation -> validation_state tzresult Lwt.t

  (** The last step in a block validation sequence. It produces the
      context that will be used as input for the validation of its
      successor block candidates. *)
  val finalize_block:
    validation_state -> validation_result tzresult Lwt.t

  (** The list of remote procedures exported by this implementation *)
  val rpc_services: rpc_context RPC.directory

  val configure_sandbox:
    Context.t -> Data_encoding.json option -> Context.t tzresult Lwt.t

end

(** Takes a version hash, a list of OCaml components in compilation
    order. The last element must be named [protocol] and respect the
    [protocol.ml] interface. Tries to compile it and returns true
    if the operation was successful. *)
val compile: Protocol_hash.t -> Protocol.t -> bool Lwt.t

(** Activates a given protocol version from a given context. This
    means that the context used for the next block will use this
    version (this is not an immediate change). The version must have
    been previously compiled successfully. *)
val activate: Context.t -> Protocol_hash.t -> Context.t Lwt.t

(** Fork a test network. The forkerd network will use the current block
    as genesis, and [protocol] as economic protocol. The network will
    be destroyed when a (successor) block will have a timestamp greater
    than [expiration]. The protocol must have been previously compiled
    successfully. *)
val fork_test_network:
  Context.t -> protocol:Protocol_hash.t -> expiration:Time.t -> Context.t Lwt.t
end
end

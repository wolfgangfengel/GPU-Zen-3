// cyCodeBase by Cem Yuksel
// [www.cemyuksel.com]
//-------------------------------------------------------------------------------
///
/// \file		cyMatrix4.h 
/// \author		Cem Yuksel
/// \version	1.5
/// \date		October 11, 2015
///
/// \brief 4x4 matrix class
///
//-------------------------------------------------------------------------------

#ifndef _CY_MATRIX4_H_INCLUDED_
#define _CY_MATRIX4_H_INCLUDED_

//-------------------------------------------------------------------------------

#include "cyPoint.h"

//-------------------------------------------------------------------------------

/// 4x4 matrix class.
/// Its data stores 16-value array of column-major matrix elements.
/// I chose column-major format to be compatible with OpenGL
/// You can use cyMatrix4f with cyPoint3f and cyPoint4f
/// to transform 3D and 4D points.
/// Both post-multiplication and pre-multiplication are supported.

class cyMatrix4f
{
	
	friend cyMatrix4f operator+( const float, const cyMatrix4f & );			///< add a value to a matrix
	friend cyMatrix4f operator-( const float, const cyMatrix4f & );			///< subtract the matrix from a value
	friend cyMatrix4f operator*( const float, const cyMatrix4f & );			///< multiple matrix by a value
	friend cyMatrix4f Inverse( cyMatrix4f &m ) { return m.GetInverse(); }	///< return the inverse of the matrix
	friend cyPoint3f  operator*( const cyPoint3f &, const cyMatrix4f & );	///< pre-multiply with a 3D point
	friend cyPoint4f  operator*( const cyPoint4f &, const cyMatrix4f & );	///< pre-multiply with a 4D point

public:

	/// elements of the matrix are column-major as in OpenGL
	float data[16];


	//////////////////////////////////////////////////////////////////////////
	///@name Constructors

	cyMatrix4f() {
		data[0] = data[1] = data[2] = 0;
		data[3] = data[4] = data[5] = 0;
		data[6] = data[7] = data[8] = 0;
		data[9] = data[10] = data[11] = 0;
	}																										///< Default constructor
	cyMatrix4f( const cyMatrix4f &matrix ) { for ( int i=0; i<16; i++ ) data[i]=matrix.data[i]; }						///< Copy constructor
	cyMatrix4f( bool identity ) { identity ? SetIdentity() : Zero(); }													///< Initialize the matrix as identity matrix or zero matrix
	cyMatrix4f( const float *array ) { Set(array); }																	///< Initialize the matrix using an array of 9 values
	cyMatrix4f( const cyPoint3f &x, const cyPoint3f &y, const cyPoint3f &z, const cyPoint3f &pos ) { Set(x,y,z,pos); }	///< Initialize the matrix using x,y,z vectors and coordinate center
	cyMatrix4f( const cyPoint4f &x, const cyPoint4f &y, const cyPoint4f &z, const cyPoint4f &w   ) { Set(x,y,z,w);   }	///< Initialize the matrix using x,y,z vectors as columns
	cyMatrix4f( const cyPoint3f &pos, const cyPoint3f &normal, const cyPoint3f &dir ) { Set(pos,normal,dir); }			///< Initialize the matrix using position, normal, and approximate x direction


	//////////////////////////////////////////////////////////////////////////
	///@name Set & Get Methods

	/// Set all the values as zero
	void Zero() { for ( int i=0; i<16; i++ ) data[ i ] = 0; }
	/// Set Matrix using an array of 16 values
	void Set( const float *array ) { for ( int i=0; i<16; i++ ) data[ i ] = array[ i ]; } 
	/// Set matrix using x,y,z vectors and coordinate center
	void Set( const cyPoint3f &x, const cyPoint3f &y, const cyPoint3f &z, const cyPoint3f &pos );
	/// Set matrix using x,y,z,w vectors
	void Set( const cyPoint4f &x, const cyPoint4f &y, const cyPoint4f &z, const cyPoint4f &w );
	/// Set matrix using position, normal, and approximate x direction
	void Set( const cyPoint3f &pos, const cyPoint3f &normal, const cyPoint3f &dir );
	/// Converts the matrix to an identity matrix
	void SetIdentity() { for(int i=0; i<16; i++) data[i]=(i%5==0) ? 1.0f : 0.0f; }
	/// Set view matrix using position, target and approximate up vector
	void SetView( const cyPoint3f &pos, const cyPoint3f &target, const cyPoint3f &up );
	/// Set matrix using normal and approximate x direction
	void SetNormal(const cyPoint3f &normal, const cyPoint3f &dir );
	/// Set as rotation matrix around x axis in radians
	void SetRotationX( float theta ) { SetRotation( cyPoint3f(1,0,0), theta ); }
	/// Set as rotation matrix around y axis in radians
	void SetRotationY( float theta ) { SetRotation( cyPoint3f(0,1,0), theta ); }
	/// Set as rotation matrix around z axis in radians
	void SetRotationZ( float theta ) { SetRotation( cyPoint3f(0,0,1), theta ); }
	/// Set a rotation matrix about the given axis by angle theta
	void SetRotation( const cyPoint3f &axis, float theta );
	/// Set a rotation matrix about the given axis by cos and sin of angle theta
	void SetRotation( const cyPoint3f &axis, float cosTheta, float sinTheta );
	/// Set a rotation matrix that sets <from> unit vector to <to> unit vector
	void SetRotation( const cyPoint3f &from, const cyPoint3f &to );
	/// Sets a uniform scale matrix
	void SetScale( float uniformScale ) { SetScale(uniformScale,uniformScale,uniformScale); }
	/// Sets a scale matrix
	void SetScale( float scaleX, float scaleY, float scaleZ );
	/// Sets a scale matrix
	void SetScale( const cyPoint3f &scale ) { SetScale(scale.x,scale.y,scale.z); }
	/// Sets a translation matrix with no rotation or scale
	void SetTrans( const cyPoint3f &move );
	/// Adds a translation to the matrix
	void AddTrans( const cyPoint3f &move );
	/// Sets the translation component of the matrix
	void SetTransComponent( const cyPoint3f &move );
	/// Set a project matrix with field of view in radians
	void SetPerspective( float fov, float aspect, float znear, float zfar ) { SetPerspectiveTan(tanf(fov*0.5f),aspect,znear,zfar); }
	/// Set a project matrix with the tangent of the half field of view (tan_fov_2)
	void SetPerspectiveTan( float tan_fov_2, float aspect, float znear, float zfar );

	// Get Row and Column
	cyPoint4f GetRow( int row ) const { return cyPoint4f( data[row], data[row+4], data[row+8], data[row+12] ); }
	void	  GetRow( int row, cyPoint4f &p ) const { p.Set( data[row], data[row+4], data[row+8], data[row+12] ); }
	void	  GetRow( int row, float *array ) const { array[0]=data[row]; array[1]=data[row+4]; array[2]=data[row+8]; array[3]=data[row+12]; }
	cyPoint4f GetColumn( int col ) const { return cyPoint4f( &data[col*4] ); }
	void	  GetColumn( int col, cyPoint4f &p ) const { p.Set( &data[col*4] ); }
	void	  GetColumn( int col, float *array ) const { array[0]=data[col*4]; array[1]=data[col*4+1]; array[2]=data[col*4+2]; array[3]=data[col*4+3]; }

	// This method can be used for converting the 3x3 portion of the matrix into a cyMatrix3
	void GetSubMatrix3data( float mdata[9] ) const
	{
		mdata[0] = data[0];
		mdata[1] = data[1];
		mdata[2] = data[2];
		mdata[3] = data[4];
		mdata[4] = data[5];
		mdata[5] = data[6];
		mdata[6] = data[8];
		mdata[7] = data[9];
		mdata[8] = data[10];
	}

	// This method can be used for converting the 2x2 portion of the matrix into a cyMatrix2
	void GetSubMatrix2data( float mdata[4] ) const
	{
		mdata[0] = data[0];
		mdata[1] = data[1];
		mdata[2] = data[4];
		mdata[3] = data[5];
	}


	//////////////////////////////////////////////////////////////////////////
	///@name Overloaded Operators

	const cyMatrix4f &operator=( const cyMatrix4f & );	///< assign matrix

	// Overloaded comparison operators 
	bool operator==( const cyMatrix4f & ) const;		///< compare equal
	bool operator!=( const cyMatrix4f &right ) const { return ! ( *this == right ); } ///< compare not equal

	// Overloaded subscript operators
	float& operator()( int row, int column );					///< subscript operator
	float& operator[](int i) { return data[i]; }				///< subscript operator
	const float& operator()( int row, int column ) const;		///< constant subscript operator
	const float& operator[](int i) const { return data[i]; }	///< constant subscript operator

	// Unary operators
	cyMatrix4f operator - () const;	///< negative matrix

	// Binary operators
	cyMatrix4f operator + ( const cyMatrix4f & ) const;	///< add two Matrices
	cyMatrix4f operator - ( const cyMatrix4f & ) const;	///< subtract one cyMatrix4f from an other
	cyMatrix4f operator * ( const cyMatrix4f & ) const;	///< multiply a matrix with an other
	cyMatrix4f operator + ( const float ) const;		///< add a value to a matrix
	cyMatrix4f operator - ( const float ) const;		///< subtract a value from a matrix
	cyMatrix4f operator * ( const float ) const;		///< multiple matrix by a value
	cyMatrix4f operator / ( const float ) const;		///< divide matrix by a value;
	cyPoint3f operator * ( const cyPoint3f& p) const;
	cyPoint4f operator * ( const cyPoint4f& p) const;

	// Assignment operators
	void	operator +=	( const cyMatrix4f & );	///< add two Matrices modify this
	void	operator -=	( const cyMatrix4f & );	///< subtract one cyMatrix4f from an other modify this matrix
	void	operator *=	( const cyMatrix4f & );	///< multiply a matrix with an other modify this matrix
	void	operator +=	( const float );		///< add a value to a matrix modify this
	void	operator -=	( const float );		///< subtract a value from a matrix modify this matrix
	void	operator *=	( const float );		///< multiply a matrix with a value modify this matrix
	void	operator /=	( const float );		///< divide the matrix by a value modify the this matrix

	//////////////////////////////////////////////////////////////////////////
	///@name Other Public Methods

	void SetTranspose();															///< Transpose this matrix
	void GetTranspose( cyMatrix4f &m ) const;										///< return Transpose of this matrix
	cyMatrix4f GetTranspose() const { cyMatrix4f t; GetTranspose(t); return t; }	///< return Transpose of this matrix

	float GetDeterminant() const;	///< Get the determinant of this matrix

	void Invert() { cyMatrix4f inv; GetInverse(inv); *this=inv; }					///< Invert this matrix
	void GetInverse( cyMatrix4f &inverse ) const;									///< Get the inverse of this matrix
	cyMatrix4f GetInverse() const { cyMatrix4f inv; GetInverse(inv); return inv; }	///< Get the inverse of this matrix

	bool IsCloseToIdentity( float tollerance=0.001f ) const;		///< Returns if the matrix is close to identity. Closeness is determined by the tollerance parameter.


	//////////////////////////////////////////////////////////////////////////
	///@name Static Methods

	/// Returns an identity matrix
	static cyMatrix4f MatrixIdentity() { cyMatrix4f m; m.SetIdentity(); return m; }
	/// Returns a view matrix using position, target and approximate up vector
	static cyMatrix4f MatrixView( const cyPoint3f &pos, const cyPoint3f &target, cyPoint3f &up ) { cyMatrix4f m; m.SetView(pos,target,up); return m; }
	/// Returns a matrix using normal, and approximate x direction
	static cyMatrix4f MatrixNormal(const cyPoint3f &normal, cyPoint3f &dir ) { cyMatrix4f m; m.SetNormal(normal,dir); return m; }
	/// Returns a rotation matrix around x axis in radians
	static cyMatrix4f MatrixRotationX( float theta ) { cyMatrix4f m; m.SetRotationX(theta); return m; }
	/// Returns a rotation matrix around y axis in radians
	static cyMatrix4f MatrixRotationY( float theta ) { cyMatrix4f m; m.SetRotationY(theta); return m; }
	/// Returns a rotation matrix around z axis in radians
	static cyMatrix4f MatrixRotationZ( float theta ) { cyMatrix4f m; m.SetRotationZ(theta); return m; }
	/// Returns a rotation matrix about the given axis by angle theta in radians
	static cyMatrix4f MatrixRotation( const cyPoint3f &axis, float theta ) { cyMatrix4f m; m.SetRotation(axis,theta); return m; }
	/// Returns a rotation matrix about the given axis by cos and sin of angle theta
	static cyMatrix4f MatrixRotation( const cyPoint3f &axis, float cosTheta, float sinTheta ) { cyMatrix4f m; m.SetRotation(axis,cosTheta,sinTheta); return m; }
	/// Returns a rotation matrix that sets <from> unit vector to <to> unit vector
	static cyMatrix4f MatrixRotation( const cyPoint3f &from, const cyPoint3f &to ) { cyMatrix4f m; m.SetRotation(from,to); return m; }
	/// Returns a uniform scale matrix
	static cyMatrix4f MatrixScale( float uniformScale ) { cyMatrix4f m; m.SetScale(uniformScale); return m; }
	/// Returns a scale matrix
	static cyMatrix4f MatrixScale( float scaleX, float scaleY, float scaleZ ) { cyMatrix4f m; m.SetScale(scaleX,scaleY,scaleZ); return m; }
	/// Returns a scale matrix
	static cyMatrix4f MatrixScale( const cyPoint3f &scale ) { cyMatrix4f m; m.SetScale(scale); return m; }
	/// Returns a translation matrix with no rotation or scale
	static cyMatrix4f MatrixTrans( const cyPoint3f &move ) { cyMatrix4f m; m.SetTrans(move); return m; }
	/// Returns a project matrix with field of view in radians
	static cyMatrix4f MatrixPerspective( float fov, float aspect, float znear, float zfar ) { cyMatrix4f m; m.SetPerspective(fov,aspect,znear,zfar); return m; }
	/// Returns a project matrix with the tangent of the half field of view (tan_fov_2)
	static cyMatrix4f MatrixPerspectiveTan( float tan_fov_2, float aspect, float znear, float zfar ) { cyMatrix4f m; m.SetPerspectiveTan(tan_fov_2,aspect,znear,zfar); return m; }

	static cyMatrix4f lookAt( cyPoint3f const& eye, cyPoint3f const& center, cyPoint3f const& up )
	{
		cyPoint3f f = (center - eye).GetNormalized();
		cyPoint3f u = (up).GetNormalized();
		cyPoint3f s = (f.Cross(u)).GetNormalized();
		u = s.Cross(f);

		cyMatrix4f Result = cyMatrix4f::MatrixIdentity();
		Result.data[0] = s.x; Result[1] = u.x; Result[2] = -f.x; Result[12] = -s.Dot(eye);
		Result.data[4] = s.y; Result[5] = u.y; Result[6] = -f.y; Result[13] = -u.Dot(eye);
		Result.data[8] = s.z; Result[9] = u.z; Result[10] = -f.z; Result[14] = f.Dot(eye);

		return Result;
	}

	static cyMatrix4f ortho(float const& left, float const& right, float const& bottom, float const& top)
	{
		cyMatrix4f Result = cyMatrix4f::MatrixIdentity();
		Result.data[0] = 2.0f / (right - left);
		Result.data[5] = 2.0f / (top - bottom);
		Result.data[10] = -1.0f;
		Result.data[12] = -(right + left) / (right - left);
		Result.data[13] = -(top + bottom) / (top - bottom);
		return Result;
	}

	static cyMatrix4f ortho(float const& left, float const& right, float const& bottom, float const& top, float const& zNear, float const& zFar)
	{
		cyMatrix4f Result = cyMatrix4f::MatrixIdentity();
		Result.data[0] = 2.0f / (right - left);
		Result.data[5] = 2.0f / (top - bottom);
		Result.data[10] = -2.0f / (zFar - zNear);
		Result.data[12] = -(right + left) / (right - left);
		Result.data[13] = -(top + bottom) / (top - bottom);
		Result.data[14] = -(zFar + zNear) / (zFar - zNear);
		return Result;
	}

	void translate(cyPoint3f const& v)
	{
		//cyMatrix4f Result;
		//for (int i = 0; i < 16; i++)
		//	Result.data[i] = data[i];
		
		data[12] += data[0] * v[0] + data[4] * v[1] +  data[8] * v[2];// + data[12];
		data[13] += data[1] * v[0] + data[5] * v[1] +  data[9] * v[2];// + data[13];
		data[14] += data[2] * v[0] + data[6] * v[1] + data[10] * v[2];// + data[14];
		data[15] += data[3] * v[0] + data[7] * v[1] + data[11] * v[2];// + data[15];
		//return Result;

		//for (int i = 0; i < 16; i++)
		//	data[i] = Result.data[i];
	}

	/////////////////////////////////////////////////////////////////////////////////
};

//-------------------------------------------------------------------------------

namespace cy {
	typedef cyMatrix4f Matrix4f;
}

//-------------------------------------------------------------------------------

/// Set Matrix using x,y,z vectors and coordinate center
inline void cyMatrix4f::Set( const cyPoint3f &x, const cyPoint3f &y, const cyPoint3f &z, const cyPoint3f &pos )
{
	x.GetValue( &data[ 0] );	data[ 3]=0;
	y.GetValue( &data[ 4] );	data[ 7]=0;
	z.GetValue( &data[ 8] );	data[11]=0;
	pos.GetValue( &data[12] );	data[15]=1;
}

//-------------------------------------------------------------------------------

/// Set Matrix using x,y,z,w vectors
inline void cyMatrix4f::Set( const cyPoint4f &x, const cyPoint4f &y, const cyPoint4f &z, const cyPoint4f &w )
{
	x.GetValue( &data[ 0] );
	y.GetValue( &data[ 4] );
	z.GetValue( &data[ 8] );
	w.GetValue( &data[12] );
}

//-------------------------------------------------------------------------------

/// Set Matrix using position, normal, and approximate x direction
inline void cyMatrix4f::Set( const cyPoint3f &pos, const cyPoint3f &normal, const cyPoint3f &dir )
{
	cyPoint3f y = normal.Cross(dir);
	y.Normalize();
	cyPoint3f newdir = y.Cross(normal);
	Set( newdir, y, normal, pos );
	
}

//-------------------------------------------------------------------------------

/// Set View Matrix using position, target and approximate up vector
inline void cyMatrix4f::SetView( const cyPoint3f &pos, const cyPoint3f &target, const cyPoint3f &up )
{
	cyPoint3f f = target - pos;
	f.Normalize();
	cyPoint3f s = f.Cross(up);
	s.Normalize();
	cyPoint3f u = s.Cross(f);

	cyMatrix4f m;
	m.SetIdentity();
	m.data[ 0]=s.x;	m.data[ 1]=u.x;	m.data[ 2]=-f.x;
	m.data[ 4]=s.y;	m.data[ 5]=u.y;	m.data[ 6]=-f.y;
	m.data[ 8]=s.z;	m.data[ 9]=u.z;	m.data[10]=-f.z;

	cyMatrix4f t;
	t.SetIdentity();
	t.data[12] = - pos.x;
	t.data[13] = - pos.y;
	t.data[14] = - pos.z;

	*this = m * t;

}

//-------------------------------------------------------------------------------

/// Set Matrix using position, normal, and approximate x direction
inline void cyMatrix4f::SetNormal( const cyPoint3f &normal, const cyPoint3f &dir )
{
	cyPoint3f y = normal.Cross(dir);
	y.Normalize();
	cyPoint3f newdir = y.Cross(normal);
	Set( newdir, y, normal, cyPoint3f(0,0,0) );
}

//-------------------------------------------------------------------------------

/// Set a rotation matrix about the given axis by angle theta
inline void cyMatrix4f::SetRotation( const cyPoint3f &axis, float theta )
{
	float c = (float) cos(theta);
	if ( c == 1 ) {
		SetIdentity();
		return;
	}
	float s = (float) sin(theta);
	SetRotation(axis,c,s);
}

//-------------------------------------------------------------------------------

/// Set a rotation matrix that sets <from> unit vector to <to> unit vector
inline void cyMatrix4f::SetRotation( const cyPoint3f &from, const cyPoint3f &to )
{
	float c = from.Dot(to);
	if ( c > 0.999999 ) {
		SetIdentity();
		return;
	}
	float s = (float) sqrtf( 1 - c * c );
	cyPoint3f axis = from.Cross(to);
	SetRotation(axis,c,s);
}

//-------------------------------------------------------------------------------

/// Set a rotation matrix about the given axis by cos and sin angle theta
inline void cyMatrix4f::SetRotation( const cyPoint3f &axis, float c, float s )
{
	if ( c == 1 ) {
		SetIdentity();
		return;
	}

	float t = 1 - c;
	float tx = t * axis.x;
	float ty = t * axis.y;
	float tz = t * axis.z;
	float txy = tx * axis.y;
	float txz = tx * axis.z;
	float tyz = ty * axis.z;
	float sx = s * axis.x;
	float sy = s * axis.y;
	float sz = s * axis.z;
	data[ 0] = tx * axis.x + c;
	data[ 1] = txy + sz;
	data[ 2] = txz - sy;
	data[ 3] = 0;
	data[ 4] = txy - sz;
	data[ 5] = ty * axis.y + c;
	data[ 6] = tyz + sx;
	data[ 7] = 0;
	data[ 8] = txz + sy;
	data[ 9] = tyz - sx;
	data[10] = tz * axis.z + c;
	data[11] = 0;
	data[12] = 0;
	data[13] = 0;
	data[14] = 0;
	data[15] = 1;
}

//-------------------------------------------------------------------------------

/// Sets a scale matrix
inline void cyMatrix4f::SetScale( float scaleX, float scaleY, float scaleZ )
{
	data[ 0] = scaleX;
	data[ 1] = 0;
	data[ 2] = 0;
	data[ 3] = 0;
	data[ 4] = 0;
	data[ 5] = scaleY;
	data[ 6] = 0;
	data[ 7] = 0;
	data[ 8] = 0;
	data[ 9] = 0;
	data[10] = scaleZ;
	data[11] = 0;
	data[12] = 0;
	data[13] = 0;
	data[14] = 0;
	data[15] = 1;
}

//-------------------------------------------------------------------------------

/// Sets the translation component of the matrix
inline void cyMatrix4f::SetTrans( const cyPoint3f &move )
{
	for(int i=0; i<12; i++) data[i]=(i%5==0) ? 1.0f : 0.0f;
	data[12] = move.x;
	data[13] = move.y;
	data[14] = move.z;
	data[15] = 1;
}

//-------------------------------------------------------------------------------

/// Adds a translation to the matrix
inline void cyMatrix4f::AddTrans( const cyPoint3f &move )
{
	data[12] += move.x;
	data[13] += move.y;
	data[14] += move.z;
}

//-------------------------------------------------------------------------------

/// Sets the translation component of the matrix
inline void cyMatrix4f::SetTransComponent( const cyPoint3f &move )
{
	data[12] = move.x;
	data[13] = move.y;
	data[14] = move.z;
}

//-------------------------------------------------------------------------------

/// Set a project matrix with field of view in radians
inline void cyMatrix4f::SetPerspectiveTan( float tan_fov_2, float aspect, float znear, float zfar )
{
	float yScale = 1.0f / tan_fov_2;
	float xScale = yScale / aspect;
	float zdif = znear - zfar;
	data[ 0] = xScale;
	data[ 1] = 0;
	data[ 2] = 0;
	data[ 3] = 0;
	data[ 4] = 0;
	data[ 5] = yScale;
	data[ 6] = 0;
	data[ 7] = 0;
	data[ 8] = 0;
	data[ 9] = 0;
	data[10] = (zfar + znear) / zdif;
	data[11] = -1;
	data[12] = 0;
	data[13] = 0;
	data[14] = (2*zfar*znear) / zdif;
	data[15] = 0;
}

//-------------------------------------------------------------------------------
// Overloaded Operators
//-------------------------------------------------------------------------------

inline cyPoint3f cyMatrix4f::operator * ( const cyPoint3f& p) const
{
	return cyPoint3f(	p.x * data[ 0] + p.y * data[ 4] + p.z * data[ 8] + data[12],
						p.x * data[ 1] + p.y * data[ 5] + p.z * data[ 9] + data[13],
						p.x * data[ 2] + p.y * data[ 6] + p.z * data[10] + data[14] );
}

//-------------------------------------------------------------------------------

inline cyPoint4f cyMatrix4f::operator * ( const cyPoint4f& p) const
{
	return cyPoint4f(	p.x * data[ 0] + p.y * data[ 4] + p.z * data[ 8] + p.w * data[12],
						p.x * data[ 1] + p.y * data[ 5] + p.z * data[ 9] + p.w * data[13],
						p.x * data[ 2] + p.y * data[ 6] + p.z * data[10] + p.w * data[14],
						p.x * data[ 3] + p.y * data[ 7] + p.z * data[11] + p.w * data[15] );
}

//-------------------------------------------------------------------------------

/// Overloaded assignment operator
/// const return avoids ( a1 = a2 ) = a3
inline const cyMatrix4f& cyMatrix4f::operator =( const cyMatrix4f &right )
{
	for ( int i = 0; i < 16; i++ ) data[i] = right.data[ i ];		// copy array into object
	return *this;	// enables x = y = z;
}

//-------------------------------------------------------------------------------

/// Determine if two arrays are equal and
/// return true, otherwise return false.
inline bool cyMatrix4f::operator ==( const cyMatrix4f &right ) const
{
	for ( int i = 0; i < 16; i++ ) {
		if ( data[ i ] != right.data[ i ] ) {
			return false;		// arrays are not equal
		}
	}
	return true;				// arrays are equal
}

//-------------------------------------------------------------------------------

/// Overloaded unary minus operator
/// negative of cyMatrix4f
inline cyMatrix4f cyMatrix4f::operator -() const
{
	cyMatrix4f buffer; // create a temp cyMatrix4f object not to change this

	for ( int i = 0; i < 16; i++ )
		buffer.data[ i ] = - data[ i ];

	// return temporary object not to change this
	return buffer;			// value return; not a reference return
}

//-------------------------------------------------------------------------------

/// Overloaded addition operator
/// add a fixed value to the matrix
inline cyMatrix4f cyMatrix4f::operator +( const float value ) const
{
	cyMatrix4f buffer; // create a temp cyMatrix4f object not to change this
	
	for ( int i = 0; i < 16; i++ )
		buffer.data[ i ] = data[ i ] + value;	// add value to all member of the matrix
	
	// return temporary object not to change this
	return buffer;			// value return; not a reference return
}

//-------------------------------------------------------------------------------

/// Overloaded addition operator
/// add two matrices
inline cyMatrix4f cyMatrix4f::operator +( const cyMatrix4f &right ) const
{
	cyMatrix4f buffer;	// create a temp cyMatrix4f object not to change this
	
	for ( int i = 0; i < 16; i++ )
		buffer.data[ i ] = data[ i ] + right.data[ i ];
	
	// return temporary object not to change this
	return buffer;			// value return; not a reference return
}

//-------------------------------------------------------------------------------

/// Overloaded addition operator
/// add a fixed value to the matrix modify matrix
inline void cyMatrix4f::operator +=( const float value )
{
	for ( int i = 0; i < 16; i++ )
		data[ i ] = data[ i ] + value;	// add value to all member of the matrix
}

//-------------------------------------------------------------------------------

/// Overloaded addition operator
/// add two matrices modify this matrix
inline void cyMatrix4f::operator +=( const cyMatrix4f &right )
{
	for ( int i = 0; i < 16; i++ )
		data[ i ] = data[ i ] + right.data[ i ];
}

//-------------------------------------------------------------------------------

/// Overloaded subtraction operator
/// subtract a fixed value from a cyMatrix4f
inline cyMatrix4f cyMatrix4f::operator -( const float value ) const
{
	cyMatrix4f buffer; // create a temp cyMatrix4f object not to change this
	
	for ( int i = 0; i < 16; i++ )
		buffer.data[ i ] = data[ i ] - value;	// subtract a value from all member of the matrix
	
	// return temporary object not to change this
	return buffer;			// value return; not a reference return
}

//-------------------------------------------------------------------------------

/// Overloaded subtraction operator
/// subtract a matrix right from this
inline cyMatrix4f cyMatrix4f::operator -( const cyMatrix4f &right ) const
{
	cyMatrix4f buffer;	// create a temp cyMatrix4f object not to change this
	
	for ( int i = 0; i < 16; i++ )
		buffer.data[ i ] = data[ i ] - right.data[ i ];
	
	// return temporary object not to change this
	return buffer;			// value return; not a reference return
}

//-------------------------------------------------------------------------------

/// Overloaded subtraction operator
/// subtract a fixed value from a cyMatrix4f modify this matrix
inline void cyMatrix4f::operator -=( const float value )
{
	for ( int i = 0; i < 16; i++ )
		data[ i ] = data[ i ] - value;	// subtract a value from all member of the matrix
}

//-------------------------------------------------------------------------------

/// Overloaded subtraction operator
/// subtract a matrix right from this modify this matrix
inline void cyMatrix4f::operator -=( const cyMatrix4f &right )
{
	for ( int i = 0; i < 16; i++ )
		data[ i ] = data[ i ] - right.data[ i ];
}

//-------------------------------------------------------------------------------

/// Overloaded multiplication operator
/// Multiply a matrix with a value
inline cyMatrix4f cyMatrix4f::operator *( const float value ) const
{
	cyMatrix4f buffer;	// create a temp cyMatrix4f object not to change this
	
	for ( int i = 0; i < 16; i++ )
		buffer.data[ i ] = data[ i ] * value;
	
	// return temporary object not to change this
	return buffer;			// value return; not a reference return
}

//-------------------------------------------------------------------------------

/// Overloaded multiplication operator
/// Multiply two matrices 
inline cyMatrix4f cyMatrix4f::operator *( const cyMatrix4f &right ) const
{
	cyMatrix4f buffer;  // a matrix of (m x k)
	
	for ( int i = 0; i < 4; i++ ) {
		for ( int k = 0; k < 4; k++ ) {
			buffer.data[ i + 4 * k ] = 0;
			for ( int j = 0; j < 4; j++ ) {
				buffer.data[ i + 4 * k ] += data[ i + 4 * j ] * right.data[ j + 4 * k ];
			}
		}
	}
	
	return buffer;
}

//-------------------------------------------------------------------------------

/// Overloaded multiplication operator
/// Multiply a matrix with a value modify this matrix
inline void cyMatrix4f::operator *=( const float value )
{
	for ( int i = 0; i < 16; i++ )
		data[ i ] = data[ i ] * value;
}

//-------------------------------------------------------------------------------

/// Overloaded multiplication operator
/// Multiply two matrices modify this matrix
inline void cyMatrix4f::operator *=( const cyMatrix4f &right )
{
	cyMatrix4f buffer;  // a matrix of (m x k)
	
	for ( int i = 0; i < 4; i++ ) {
		for ( int k = 0; k < 4; k++ ) {
			buffer.data[ i + 4 * k ] = 0;
			for ( int j = 0; j < 4; j++ ) {
				buffer.data[ i + 4 * k ] += data[ i + 4 * j ] * right.data[ j + 4 * k ];
			}
		}
	}
	
	*this = buffer;	// using a buffer to calculate the result
	//then copy buffer to this
}

//-------------------------------------------------------------------------------

/// Overloaded division operator
/// Divide the matrix by value
inline cyMatrix4f cyMatrix4f::operator /( const float value ) const
{
	if ( value == 0 ) return *this;
	return operator * ( (float) 1.0 / value );
}

//-------------------------------------------------------------------------------

/// Overloaded division operator
/// Divide the matrix by value
inline void cyMatrix4f::operator /=( const float value )
{
	if ( value == 0 ) return;
	
	for ( int i = 0; i < 16; i++ )
		data[ i ] = data[ i ] / value;
	
}

//-------------------------------------------------------------------------------

/// Overloaded subscript operator for non-const cyMatrix4f
/// reference return creates an lvalue
inline float& cyMatrix4f::operator ()( int row, int column )
{
	return data[ column * 4 + row ];	// reference return
}

//-------------------------------------------------------------------------------

/// Overloaded subscript operator for const cyMatrix4f
/// const reference return creates an rvalue
inline const float& cyMatrix4f::operator ()( int row, int column ) const
{
	return data[ column * 4 + row ];	// const reference return
}


//-------------------------------------------------------------------------------
//-------------------------------------------------------------------------------

/// Transpose of this matrix
inline void cyMatrix4f::SetTranspose()
{
	float temp;

    for ( int i = 1; i < 4; i++ ) {
		for ( int j = 0; j < i; j++ ) {
			temp = data[ i * 4 + j ];
			data[ i * 4 + j ] = data[ j * 4 + i ];
			data[ j * 4 + i ] = temp;
		}
    }
}

//-------------------------------------------------------------------------------

inline void cyMatrix4f::GetTranspose(cyMatrix4f &m) const
{
	m.data[ 0] = data[ 0];
	m.data[ 1] = data[ 4];
	m.data[ 2] = data[ 8];
	m.data[ 3] = data[12];
	m.data[ 4] = data[ 1];
	m.data[ 5] = data[ 5];
	m.data[ 6] = data[ 9];
	m.data[ 7] = data[13];
	m.data[ 8] = data[ 2];
	m.data[ 9] = data[ 6];
	m.data[10] = data[10];
	m.data[11] = data[14];
	m.data[12] = data[ 3];
	m.data[13] = data[ 7];
	m.data[14] = data[11];
	m.data[15] = data[15];
}

//-------------------------------------------------------------------------------

inline float cyMatrix4f::GetDeterminant() const
{
	return	data[12]*(  data[ 9]*(data[ 6]*data[ 3]-data[ 2]*data[ 7]) +data[ 5]*(-(data[10]*data[ 3])+data[ 2]*data[11])-data[ 1]*(-(data[10]*data[ 7])+data[ 6]*data[11])) + 
			data[ 8]*(-(data[13]*(data[ 6]*data[ 3]-data[ 2]*data[ 7]))-data[ 5]*(-(data[14]*data[ 3])+data[ 2]*data[15])+data[ 1]*(-(data[14]*data[ 7])+data[ 6]*data[15])) - 
			data[ 4]*(-(data[13]*(data[10]*data[ 3]-data[ 2]*data[11]))-data[ 9]*(-(data[14]*data[ 3])+data[ 2]*data[15])+data[ 1]*(-(data[14]*data[11])+data[10]*data[15])) + 
			data[ 0]*(-(data[13]*(data[10]*data[ 7]-data[ 6]*data[11]))-data[ 9]*(-(data[14]*data[ 7])+data[ 6]*data[15])+data[ 5]*(-(data[14]*data[11])+data[10]*data[15]));
}

//-------------------------------------------------------------------------------

inline void cyMatrix4f::GetInverse( cyMatrix4f &inverse ) const
{
	float a = 1.0f / GetDeterminant();

	inverse.data[ 0] = ( data[ 5]*data[10]*data[15] + data[ 9]*data[14]*data[ 7] + data[13]*data[ 6]*data[11] - data[ 5]*data[14]*data[11] - data[ 9]*data[ 6]*data[15] - data[13]*data[10]*data[ 7] ) * a;
	inverse.data[ 1] = ( data[ 1]*data[14]*data[11] + data[ 9]*data[ 2]*data[15] + data[13]*data[10]*data[ 3] - data[ 1]*data[10]*data[15] - data[ 9]*data[14]*data[ 3] - data[13]*data[ 2]*data[11] ) * a;
	inverse.data[ 2] = ( data[ 1]*data[ 6]*data[15] + data[ 5]*data[14]*data[ 3] + data[13]*data[ 2]*data[ 7] - data[ 1]*data[14]*data[ 7] - data[ 5]*data[ 2]*data[15] - data[13]*data[ 6]*data[ 3] ) * a;
	inverse.data[ 3] = ( data[ 1]*data[10]*data[ 7] + data[ 5]*data[ 2]*data[11] + data[ 9]*data[ 6]*data[ 3] - data[ 1]*data[ 6]*data[11] - data[ 5]*data[10]*data[ 3] - data[ 9]*data[ 2]*data[ 7] ) * a;
	inverse.data[ 4] = ( data[ 4]*data[14]*data[11] + data[ 8]*data[ 6]*data[15] + data[12]*data[10]*data[ 7] - data[ 4]*data[10]*data[15] - data[ 8]*data[14]*data[ 7] - data[12]*data[ 6]*data[11] ) * a;
	inverse.data[ 5] = ( data[ 0]*data[10]*data[15] + data[ 8]*data[14]*data[ 3] + data[12]*data[ 2]*data[11] - data[ 0]*data[14]*data[11] - data[ 8]*data[ 2]*data[15] - data[12]*data[10]*data[ 3] ) * a;
	inverse.data[ 6] = ( data[ 0]*data[14]*data[ 7] + data[ 4]*data[ 2]*data[15] + data[12]*data[ 6]*data[ 3] - data[ 0]*data[ 6]*data[15] - data[ 4]*data[14]*data[ 3] - data[12]*data[ 2]*data[ 7] ) * a;
	inverse.data[ 7] = ( data[ 0]*data[ 6]*data[11] + data[ 4]*data[10]*data[ 3] + data[ 8]*data[ 2]*data[ 7] - data[ 0]*data[10]*data[ 7] - data[ 4]*data[ 2]*data[11] - data[ 8]*data[ 6]*data[ 3] ) * a;
	inverse.data[ 8] = ( data[ 4]*data[ 9]*data[15] + data[ 8]*data[13]*data[ 7] + data[12]*data[ 5]*data[11] - data[ 4]*data[13]*data[11] - data[ 8]*data[ 5]*data[15] - data[12]*data[ 9]*data[ 7] ) * a;
	inverse.data[ 9] = ( data[ 0]*data[13]*data[11] + data[ 8]*data[ 1]*data[15] + data[12]*data[ 9]*data[ 3] - data[ 0]*data[ 9]*data[15] - data[ 8]*data[13]*data[ 3] - data[12]*data[ 1]*data[11] ) * a;
	inverse.data[10] = ( data[ 0]*data[ 5]*data[15] + data[ 4]*data[13]*data[ 3] + data[12]*data[ 1]*data[ 7] - data[ 0]*data[13]*data[ 7] - data[ 4]*data[ 1]*data[15] - data[12]*data[ 5]*data[ 3] ) * a;
	inverse.data[11] = ( data[ 0]*data[ 9]*data[ 7] + data[ 4]*data[ 1]*data[11] + data[ 8]*data[ 5]*data[ 3] - data[ 0]*data[ 5]*data[11] - data[ 4]*data[ 9]*data[ 3] - data[ 8]*data[ 1]*data[ 7] ) * a;
	inverse.data[12] = ( data[ 4]*data[13]*data[10] + data[ 8]*data[ 5]*data[14] + data[12]*data[ 9]*data[ 6] - data[ 4]*data[ 9]*data[14] - data[ 8]*data[13]*data[ 6] - data[12]*data[ 5]*data[10] ) * a;
	inverse.data[13] = ( data[ 0]*data[ 9]*data[14] + data[ 8]*data[13]*data[ 2] + data[12]*data[ 1]*data[10] - data[ 0]*data[13]*data[10] - data[ 8]*data[ 1]*data[14] - data[12]*data[ 9]*data[ 2] ) * a;
	inverse.data[14] = ( data[ 0]*data[13]*data[ 6] + data[ 4]*data[ 1]*data[14] + data[12]*data[ 5]*data[ 2] - data[ 0]*data[ 5]*data[14] - data[ 4]*data[13]*data[ 2] - data[12]*data[ 1]*data[ 6] ) * a;
	inverse.data[15] = ( data[ 0]*data[ 5]*data[10] + data[ 4]*data[ 9]*data[ 2] + data[ 8]*data[ 1]*data[ 6] - data[ 0]*data[ 9]*data[ 6] - data[ 4]*data[ 1]*data[10] - data[ 8]*data[ 5]*data[ 2] ) * a;
}

//-------------------------------------------------------------------------------

inline bool cyMatrix4f::IsCloseToIdentity( float tollerance ) const
{
	for(int i=0; i<16; i++ ) {
		float v = (i%5==0) ? 1.0f : 0.0f;
		if ( fabs(data[i]-v) > tollerance ) return false;
	}
	return true;
}

//-------------------------------------------------------------------------------
// friend function definitions
//-------------------------------------------------------------------------------

/// Overloaded addition operator
/// add a fixed value to the matrix
inline cyMatrix4f operator+( const float value, const cyMatrix4f &right )
{
	cyMatrix4f buffer; // create a temp cyMatrix4f object not to change right
	
	for ( int i = 0; i < 16; i++ )
		buffer.data[ i ] = right.data[ i ] + value;	// add value to all members of the matrix
	
	// return temporary object not to change right
	return buffer;			// value return; not a reference return
}

//-------------------------------------------------------------------------------

/// Overloaded subtraction operator
/// subtract the matrix from a fixed value
inline cyMatrix4f operator-( const float value, const cyMatrix4f &right )
{
	cyMatrix4f buffer; // create a temp cyMatrix4f object not to change right
	
	for ( int i = 0; i < 16; i++ )
		buffer.data[ i ] = value - right.data[ i ];	// subtract matrix from the value;
	
	// return temporary object not to change right
	return buffer;			// value return; not a reference return
}

//-------------------------------------------------------------------------------

/// Overloaded multiplication operator
/// multiply a fixed value with the matrix
inline cyMatrix4f operator*( const float value, const cyMatrix4f &right )
{
	cyMatrix4f buffer; // create a temp cyMatrix4f object not to change right
	
	for ( int i = 0; i < 16; i++ )
		buffer.data[ i ] = right.data[ i ] * value;	// multiply value to all members of the matrix
	
	// return temporary object not to change right
	return buffer;			// value return; not a reference return
}

//-------------------------------------------------------------------------------

inline cyPoint3f operator * ( const cyPoint3f& p, const cyMatrix4f &m )
{
	return cyPoint3f(	p.x * m.data[ 0] + p.y * m.data[ 1] + p.z * m.data[ 2] + m.data[ 3],
						p.x * m.data[ 4] + p.y * m.data[ 5] + p.z * m.data[ 6] + m.data[ 7],
						p.x * m.data[ 8] + p.y * m.data[ 9] + p.z * m.data[10] + m.data[11] );
}

//-------------------------------------------------------------------------------

inline cyPoint4f operator * ( const cyPoint4f& p, const cyMatrix4f &m )
{
	return cyPoint4f(	p.x * m.data[ 0] + p.y * m.data[ 1] + p.z * m.data[ 2] + p.w * m.data[ 3],
						p.x * m.data[ 4] + p.y * m.data[ 5] + p.z * m.data[ 6] + p.w * m.data[ 7],
						p.x * m.data[ 8] + p.y * m.data[ 9] + p.z * m.data[10] + p.w * m.data[11],
						p.x * m.data[12] + p.y * m.data[13] + p.z * m.data[14] + p.w * m.data[15] );
}

//-------------------------------------------------------------------------------

#endif

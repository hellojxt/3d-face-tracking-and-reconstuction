
struct fPoint{
	float x,y,z;
	__device__ fPoint(float x_, float y_, float z_){
        x = x_;
        y = y_;
        z = z_;
    }
};

__device__ fPoint Pmin(fPoint p1, fPoint p2){
	float x = p1.x;
	if (p2.x < x)
		x = p2.x;
	float y = p1.y;
	if (p2.y < y)
		y = p2.y;
	return fPoint(x,y,0);
}

__device__ fPoint Pmax(fPoint p1, fPoint p2){
	float x = p1.x;
	if (p2.x > x)
		x = p2.x;
	float y = p1.y;
	if (p2.y > y)
		y = p2.y;
	return fPoint(x,y,0);
}

__device__ float sign (fPoint p1, fPoint p2, fPoint p3)
{
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

__device__ bool PointInTriangle (fPoint pt, fPoint v1, fPoint v2, fPoint v3)
{
    float d1, d2, d3;
    bool has_neg, has_pos;

    d1 = sign(pt, v1, v2);
    d2 = sign(pt, v2, v3);
    d3 = sign(pt, v3, v1);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}

__global__  void render(float vertices[3][53215], int tris[3][105840], 
                        float colors[3][53215], int depth[WIDTH][HEIGHT], char image[WIDTH][HEIGHT][3]){
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i >= 105840){
		return;
	}
    
    
	fPoint v1(vertices[0][int(tris[0][i])],vertices[1][int(tris[0][i])],vertices[2][int(tris[0][i])]);
	fPoint v2(vertices[0][int(tris[1][i])],vertices[1][int(tris[1][i])],vertices[2][int(tris[1][i])]);
	fPoint v3(vertices[0][int(tris[2][i])],vertices[1][int(tris[2][i])],vertices[2][int(tris[2][i])]);
	fPoint c1(colors[0][int(tris[0][i])],colors[1][int(tris[0][i])],colors[2][int(tris[0][i])]);
	fPoint c2(colors[0][int(tris[1][i])],colors[1][int(tris[1][i])],colors[2][int(tris[1][i])]);
	fPoint c3(colors[0][int(tris[2][i])],colors[1][int(tris[2][i])],colors[2][int(tris[2][i])]);
	fPoint leftup = Pmin(Pmin(v1,v2),v3);
	fPoint rightdown = Pmax(Pmax(v1,v2),v3);
	for (int x = int(leftup.x);x <= int(rightdown.x);x++)
		for (int y = int(leftup.y);y <= int(rightdown.y);y++){

			if (PointInTriangle(fPoint(x,y,0),v1,v2,v3)){
				int d = int((v1.z+v2.z+v3.z)/3);
                atomicMax(&depth[y][x],d);
                if (depth[y][x] == d){
					image[y][x][0] = (c1.x+c2.x+c3.x)/3;
					image[y][x][1] = (c1.y+c2.y+c3.y)/3;
					image[y][x][2] = (c1.z+c2.z+c3.z)/3;
				}
            }
		}
	
}
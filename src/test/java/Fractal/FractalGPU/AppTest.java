package Fractal.FractalGPU;

import static org.jocl.CL.*;

import java.awt.Canvas;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GraphicsDevice;
import java.awt.GraphicsEnvironment;
import java.awt.Rectangle;
import java.awt.Toolkit;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.awt.image.VolatileImage;
import java.util.Arrays;
import java.util.Scanner;

import javax.swing.JFrame;

import org.jocl.*;

/**
 * A small JOCL sample.
 */
public class AppTest extends Canvas implements MouseListener, KeyListener
{ 
	/**
	 * The source code of the OpenCL program to execute

	 */
	static Dimension screensize= Toolkit.getDefaultToolkit().getScreenSize();
	static int sizex=(int)screensize.getWidth()/2;
	static int sizey=(int)screensize.getHeight()/2;
	static int size = (sizex<sizey)? sizex:sizey;
	private static BufferedImage buf = new BufferedImage(sizex,sizey,BufferedImage.TYPE_INT_RGB);
	private static String programSource =
			"__kernel void "+
					"sampleKernel(__global const float *meta,"
					+ "			  __global const float *a,"+
					"             __global const float *b,"+
					"             __global float *c)"+
					"{"+
					"   int gid = get_global_id(0);"+
					"	float r = 0;"+
					"	float i = 0;"+//r and i represent imag z
					"	float workingR=0;"+
					"	int rr=0;"+
					"	int ir=0;"+
					"	c[gid]=-1;"+
					"	for(int counter=0; counter<70*meta[0]; counter++){"+
					"		workingR=r;"+
					"   	r=(r*r)-(i*i);"+//stat multiplication (z*z)
					"   	i=(workingR*i)*2;"+//end mult
					"		r=r+a[gid];"+//start add z+c
					"		i=i+b[gid];"+//end add
					"		rr=floor((float)r);"+
					"		ir=floor((float)i);"+
					"		rr=fabs((float)rr);"+
					"		ir=fabs((float)ir);"+
					"   	if(rr>2||ir>2){"+
				//	"   		c[gid]=counter*(255.0f/iterations);"+
					"			c[gid]=counter*(255.0f/(70.0f*meta[0]));"+
					"   		return;"+
					"   	}"+
					"	}"+
					"   c[gid]=255;"+
					"   return;"+
					"}";
	static cl_mem memObjects[];
	static Pointer srcA;
	static Pointer srcB;
	static Pointer dst;
	static Pointer meta;
	static float[] srcArrayA;
	static float[] srcArrayB;
	static float[] dstArray;
	static float[] metaArray;
	static int n;
	static cl_context context;
	static cl_kernel kernel;
	static cl_command_queue commandQueue;
	static cl_program program;
	static Canvas thiscanvas;
	static JFrame thisframe;
	static long zoom=0;
	static float xtrans=0;
	static float ytrans=0;
	/**
	 * The entry point of this sample
	 * 
	 * @param args Not used
	 * @throws Exception 
	 */
	public static void main(String args[]) throws Exception
	{
		JFrame frame = new JFrame("Fractal");
		thisframe=frame;

		frame.setSize(new Dimension(sizex,sizey));
//		frame.setExtendedState(JFrame.MAXIMIZED_BOTH); 
//		frame.setUndecorated(true);
//		frame.setVisible(true);

		final Canvas canvas;
		canvas = new AppTest();
		thiscanvas=canvas;
		canvas.setSize(sizex, sizey);
		frame.add(canvas);
		frame.setVisible(true);
		frame.setDefaultCloseOperation(frame.EXIT_ON_CLOSE);
		frame.requestFocus();
		canvas.setBackground(Color.black);
		canvas.addMouseListener((MouseListener)canvas);
		frame.addKeyListener((KeyListener)canvas);
		thiscanvas.addKeyListener((KeyListener)canvas);
		
		gpuinit();
		long timestart = System.currentTimeMillis();
		gpuexecute(zoom,xtrans,ytrans);

		// Release kernel, program, and memory objects
		Runtime.getRuntime().addShutdownHook(new Thread()
	    {
	      public void run()
	      {
	        System.out.println("Releasing resources");
	        clReleaseMemObject(memObjects[0]);
			clReleaseMemObject(memObjects[1]);
			clReleaseMemObject(memObjects[2]);
			clReleaseKernel(kernel);
			clReleaseProgram(program);
			clReleaseCommandQueue(commandQueue);
			clReleaseContext(context);
	      }
	    });
		
		for(int i=0; i<14; i++) {
			ytrans=-0.7241127834f;
			xtrans=-0.2864591370f;
//			xtrans=0.001643721971153;
//			ytrans=0.822467633298876;
			ytrans=0.001643721971153f;
			xtrans=0.822467633298876f;
			zoom=i;
			gpuexecute(zoom,xtrans,ytrans);
		}
		

		
		System.out.println("TOTAL TIME: "+(System.currentTimeMillis()-timestart));
	}
	private static void gpuinit() {

		long numBytes[] = new long[1];
		// Create input- and output data 
		n = 1;
		srcArrayA = new float[n];
		srcArrayB = new float[n];
		dstArray = new float[n];
		for (int i=-n/2; i<n/2; i++)
		{
			srcArrayA[i+(n/2)] = 2;
			srcArrayB[i+(n/2)] = (float) (i*(2.0/(n/2)));
			dstArray[i+(n/2)] = 0;
		}
		metaArray = new float[1];
		srcA = Pointer.to(srcArrayA);
		srcB = Pointer.to(srcArrayB);
		dst = Pointer.to(dstArray);
		meta=  Pointer.to(metaArray);

		// Obtain the platform IDs and initialize the context properties
		System.out.println("Obtaining platform...");
		cl_platform_id platforms[] = new cl_platform_id[1];
		clGetPlatformIDs(platforms.length, platforms, null);
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, platforms[0]);

		// Create an OpenCL context on a GPU device
		context = clCreateContextFromType(
				contextProperties, CL_DEVICE_TYPE_GPU, null, null, null);
		if (context == null)
		{
			// If no context for a GPU device could be created,
			// try to create one for a CPU device.
			context = clCreateContextFromType(
					contextProperties, CL_DEVICE_TYPE_CPU, null, null, null);
			System.out.println("WARNING USING CPU");

			if (context == null)
			{
				System.out.println("Unable to create a context");
				return;
			}
		}

		// Enable exceptions and subsequently omit error checks in this sample
		CL.setExceptionsEnabled(true);

		// Get the list of GPU devices associated with the context
		clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, null, numBytes); 

		// Obtain the cl_device_id for the first device
		int numDevices = (int) numBytes[0] / Sizeof.cl_device_id;
		cl_device_id devices[] = new cl_device_id[numDevices];
		clGetContextInfo(context, CL_CONTEXT_DEVICES, numBytes[0],  
				Pointer.to(devices), null);

		// Create a command-queue
		commandQueue = 
				clCreateCommandQueue(context, devices[0], 0, null);

		// Allocate the memory objects for the input- and output data
		memObjects = new cl_mem[4];
		memObjects[0]= clCreateBuffer(context, 
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				Sizeof.cl_float * metaArray.length, meta, null);
		memObjects[1] = clCreateBuffer(context, 
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				Sizeof.cl_float * n, srcA, null);
		memObjects[2] = clCreateBuffer(context, 
				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				Sizeof.cl_float * n, srcB, null);
		memObjects[3] = clCreateBuffer(context, 
				CL_MEM_READ_WRITE, 
				Sizeof.cl_float * n, null, null);

		// Create the program from the source code
		program = clCreateProgramWithSource(context,
				1, new String[]{ programSource }, null, null);

		// Build the program
		clBuildProgram(program, 0, null, null, null, null);

		// Create the kernel
		kernel = clCreateKernel(program, "sampleKernel", null);

		// Set the arguments for the kernel
		clSetKernelArg(kernel, 0, 
				Sizeof.cl_mem, Pointer.to(memObjects[0]));
		clSetKernelArg(kernel, 1, 
				Sizeof.cl_mem, Pointer.to(memObjects[1]));
		clSetKernelArg(kernel, 2, 
				Sizeof.cl_mem, Pointer.to(memObjects[2]));
		clSetKernelArg(kernel, 3, 
				Sizeof.cl_mem, Pointer.to(memObjects[3]));
	}
	public static void gpuexecute(long zoom, float xtrans, float ytrans) throws Exception {
		int n = sizex*sizey;
		long global_work_size[] = new long[]{n};
		long local_work_size[] = new long[]{1};
		float[][] imagarray = new float[sizey][sizex];
		float[][] realarray = new float[sizey][sizex];
		Graphics2D g2 = (Graphics2D)buf.getGraphics();
			for(int oi=0; oi<sizey; oi++) {
				for(int ii=0; ii<sizex; ii++) {
//					imagarray[oi][ii]=(float) ((((oi-800.0)*(2.0/800))/Math.pow(10,1.0*zoom/3))+xtrans);
//					realarray[oi][ii]= (float) ((((ii-800.0)*(2.0/800))/Math.pow(10, 1.0*zoom/3))+ytrans);
					
					imagarray[oi][ii]=(float) ((((oi-(size/2))*(2.0/(size/2)))/(Math.pow(10,1.0*zoom/2)))+xtrans);
					realarray[oi][ii]= (float) ((((ii-(size/2))*(2.0/(size/2)))/(Math.pow(10, 1.0*zoom/2)))+ytrans);
				}
			}
			metaArray[0]=(int)Math.round((1.0f*zoom/2)+1);
			srcArrayA=flattenarray(realarray);
			srcArrayB=flattenarray(imagarray);
			dstArray=new float[srcArrayA.length];
			srcA = Pointer.to(srcArrayA);
			srcB = Pointer.to(srcArrayB);
			dst = Pointer.to(dstArray);
			meta = Pointer.to(metaArray);
			n=srcArrayA.length;
			clReleaseMemObject(memObjects[0]);
			clReleaseMemObject(memObjects[1]);
			clReleaseMemObject(memObjects[2]);
			clReleaseMemObject(memObjects[3]);
			memObjects[0]= clCreateBuffer(context, 
					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					Sizeof.cl_float * metaArray.length, meta, null);
			memObjects[1] = clCreateBuffer(context, 
					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					Sizeof.cl_float * n, srcA, null);
			memObjects[2] = clCreateBuffer(context, 
					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					Sizeof.cl_float * n, srcB, null);
			memObjects[3] = clCreateBuffer(context, 
					CL_MEM_READ_WRITE, 
					Sizeof.cl_float * n, null, null);
			global_work_size = new long[]{n};
			clSetKernelArg(kernel, 0, 
					Sizeof.cl_mem, Pointer.to(memObjects[0]));
			clSetKernelArg(kernel, 1, 
					Sizeof.cl_mem, Pointer.to(memObjects[1]));
			clSetKernelArg(kernel, 2, 
					Sizeof.cl_mem, Pointer.to(memObjects[2]));
			clSetKernelArg(kernel, 3, 
					Sizeof.cl_mem, Pointer.to(memObjects[3]));
			clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
					global_work_size, local_work_size, 0, null, null);   

			clEnqueueReadBuffer(commandQueue, memObjects[3], CL_TRUE, 0,
					n * Sizeof.cl_float, dst, 0, null, null);
			float[][] image = expandarray(dstArray,sizex);
			for (int y=0; y<sizey; y++) {
				for(int x=0; x<sizex; x++) {
					//				g2.setColor(new Color((int)image[y][x],(int)image[y][x],(int)image[y][x]));
					//				g2.fillOval(x, y, 2, 2);
//					buf.setRGB(x, y, colorscheme((int)image[y][x]).getRGB());
					buf.setRGB(x, y, Color.HSBtoRGB((1.0f*image[y][x]/(255.0f)),1,(1.0f*image[y][x]/(256.0f))) );
					
				}
			}

//			canvas.repaint();
			thiscanvas.paint(thiscanvas.getGraphics());
		
	}
	public static float[] flattenarray(float[][] a) {
		float[] b = new float[a.length*a[0].length];
		int shifter=0;
		for(int i=0; i<a.length; i++) {
			for(int index=0; index<a[i].length; index++) {
				b[i+index+shifter]=a[i][index];
			}
			shifter+=a[0].length-1;
		}
		return b;
	}
	public static float[][] expandarray(float[] a,int size) throws Exception{
		if(a.length%size!=0) {
			throw new Exception("invalid size");
		}
		float[][] out = new float[a.length/size][size];
		float[] temp = new float[size];
		int outerindex=0;

		for(int i=0; i<a.length; ) {
			for(int c=0; c<size; c++) {
				temp[c]=a[i];
				i++;
			}
			out[outerindex]=temp;
			outerindex++;
			temp = new float[size];
		}
		return out;
	}
	public static Color colorscheme(int a) {
		//0<=a<=255
		//Black -> Red -> Yellow -> White

		double r1,r2,g1,g2,b1,b2;
		int rf= 255;
		int bf=255;
		int gf=255;
		Color Aa = new Color(2,2,2);

		if(a==0) {
			return new Color(0,0,0);
			//BLACK
		}
		else if(a<=85) {
			//RED
			r1=(0*(1.0-(1.0*a/85)));
			g1=(0*(1.0-(1.0*a/85)));
			b1=(0*(1.0-(1.0*a/85)));
			r2=(255*(1.0*a/85));
			g2=(0*(1.0*a/85));
			b2=(0*(1.0*a/85));

			rf=(int)Math.round((r1+r2));
			bf=(int)Math.round((b1+b2));
			gf=(int)Math.round((g1+g2));


		}
		else if(a<=113) {
			a-=85;

			r1=(255*(1.0-(1.0*a/28)));
			g1=(0*(1.0-(1.0*a/28)));
			b1=(0*(1.0-(1.0*a/28)));
			r2=(245*(1.0*a/28));
			g2=(245*(1.0*a/28));
			b2=(66*(1.0*a/28));

			rf=(int)Math.round((r1+r2));
			bf=(int)Math.round((b1+b2));
			gf=(int)Math.round((g1+g2));

			//YELLOW
		}
		else if(a<=255) {
			a-=113;

			r1=(245*(1.0-(1.0*a/142)));
			g1=(245*(1.0-(1.0*a/142)));
			b1=(66*(1.0-(1.0*a/142)));
			r2=(255*(1.0*a/142));
			g2=(255*(1.0*a/142));
			b2=(255*(1.0*a/142));

			rf=(int)Math.round((r1+r2));
			bf=(int)Math.round((b1+b2));
			gf=(int)Math.round((g1+g2));

			//WHITE
		}
		else {
			return new Color(255,255,255);
		}
		return new Color(rf,bf,gf);

	}
	public void paint(Graphics g) {
		
//		if(thisframe.getWidth()!=800||thisframe.getHeight()!=800) {
//			Rectangle bounds = thisframe.getBounds();
//			int scalar=800;
//			if(bounds.getWidth()<bounds.getHeight()) {
//				scalar=(int)Math.round(bounds.getWidth());
//			}
//			else {
//				scalar=(int)Math.round(bounds.getHeight());
//			}
//			if(scalar==0) scalar=800;
//			System.out.println(bounds);
//			buf.getScaledInstance(scalar,scalar, 0);		
//			g.drawImage(buf.getScaledInstance(scalar,scalar, 0),0,0,null);
//		}
//		else {
//			g.drawImage(buf,0,0,null);
//		}
		g.drawImage(buf,0,0,null);
	}
	public void mouseClicked(MouseEvent e) {
		// TODO Auto-generated method stub
		switch(e.getButton()){
		case 1:
			System.out.println("Zooming");
			System.out.println("OLD x+"+xtrans+" y+"+ytrans+" ×"+zoom);
			System.out.println("POINTED TO "+(((e.getX()-400)/(400.0/2)))+","+((e.getY()-400)/(400.0/2)));
			System.out.println("TRANSFORM BY "+((((400-e.getX())/(400.0/2))/(1.0*Math.pow(10,1.0*zoom/3))))+","+((((400-e.getY())/(400.0/2))/(1.0*Math.pow(10,1.0*zoom/3)))));
			zoom++;
			xtrans+= (float) (((400-e.getX())/(400.0/2))/(1.0*Math.pow(10,1.0*zoom/3)));
			ytrans+= (float) (((400-e.getY())/(400.0/2))/(1.0*Math.pow(10,1.0*zoom/3)));
			
			System.out.println("NEW x+"+xtrans+" y+"+ytrans+" ×"+zoom);
			try {
				Graphics2D g2= (Graphics2D)buf.getGraphics();
				g2.setColor(Color.black);
				g2.fillRect(0, 0, 14*11, 40);
				g2.setColor(Color.white);
				g2.drawString("Loading...",10,10);
				thiscanvas.paint(thiscanvas.getGraphics());
				gpuexecute(zoom,xtrans,ytrans);
			} catch (Exception e1) {
				e1.printStackTrace();
			}
			break;
		case 2:
			System.exit(0);
			break;
		case 3: 
			zoom--;
			xtrans+= (((e.getX()-400)/(400.0/2))/(1.0*Math.pow(10,1.0*zoom/3)));
			ytrans+= (((e.getY()-400)/(400.0/2))/(1.0*Math.pow(10,1.0*zoom/3)));
			try {
				Graphics2D g2= (Graphics2D)buf.getGraphics();
				g2.setColor(Color.black);
				g2.fillRect(0, 0, 14*11, 40);
				g2.setColor(Color.white);
				g2.drawString("Loading...",10,10);
				thiscanvas.paint(thiscanvas.getGraphics());
				gpuexecute(zoom,xtrans,ytrans);
			} catch (Exception e1) {
				e1.printStackTrace();
			}
			break;
		default:
			break;
		}
			
	}
	public void mouseEntered(MouseEvent arg0) {
		// TODO Auto-generated method stub

	}
	public void mouseExited(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	public void mousePressed(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	public void mouseReleased(MouseEvent arg0) {
		// TODO Auto-generated method stub
		
	}
	public void keyPressed(KeyEvent arg0) {
		// TODO Auto-generated method stub
	}
	public void keyReleased(KeyEvent arg0) {
		// TODO Auto-generated method stub
	}
	public void keyTyped(KeyEvent e) {
		// TODO Auto-generated method stub
		System.out.println(e.getKeyChar());
		
		switch(new String(""+e.getKeyChar()).toLowerCase().charAt(0)){
		case ''://esc
			System.exit(0);
			break;
		case 'r':
			zoom=0;
			xtrans=0;
			ytrans=0;
			try {
				Graphics2D g2= (Graphics2D)buf.getGraphics();
				g2.setColor(Color.black);
				g2.fillRect(0, 0, 14*11, 40);
				g2.setColor(Color.white);
				g2.drawString("Loading...",10,10);
				thiscanvas.paint(thiscanvas.getGraphics());
				gpuexecute(zoom,xtrans,ytrans);
			} catch (Exception e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			break;
		default:
			break;
			
		}
	}

}